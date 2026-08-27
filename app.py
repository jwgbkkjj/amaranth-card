import io
import json
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from openpyxl.styles import PatternFill

st.set_page_config(page_title="AI 법인카드 전표 변환기", page_icon="💳", layout="wide")

# ==============================================================================
# [설정 1] 사내 접속 비밀번호
# ==============================================================================
PASSWORD = "sj123456!"

# ==============================================================================
# [설정 2] Google Gemini API Key 설정
# ==============================================================================
GEMINI_API_KEY = "AQ.Ab8RN6LswiAaKJaztHp9yh1qd6xFwb1nll_N79XHfJNPW5lReA"

st.title("💳 법인카드 $\\rightarrow$ 아마란스 10 전표 변환기")
st.caption("신한/우리/하나카드 자동 인식 | 7자리 계정과목 적용 | 아마란스 서식(회계단위 1000) 맞춤")

# 비밀번호 로그인
input_pw = st.text_input("접속 비밀번호를 입력하세요", type="password")
if input_pw != PASSWORD:
    if input_pw:
        st.error("비밀번호가 일치하지 않습니다.")
    st.stop()

st.success("인증 완료되었습니다.")
st.divider()

if GEMINI_API_KEY == "여기에_발급받은_API_키를_붙여넣으세요" or not GEMINI_API_KEY:
    st.error("코드 19번째 줄의 'GEMINI_API_KEY'에 실제 API 키를 입력 후 저장해주세요.")
    st.stop()

# 사이드바 설정
with st.sidebar:
    st.header("🏢 아마란스 10 기본값")
    default_year = st.text_input("전표 기본 연도", value=str(datetime.now().year))
    default_credit_account = st.text_input("대변(미지급금) 계정코드", value="2530000")
    st.info("💡 신한/우리/하나카드는 업로드 시 대변 거래처명이 자동으로 지정됩니다.")

# ------------------------------------------------------------------------------
# Pydantic 데이터 구조 및 헬퍼 함수
# ------------------------------------------------------------------------------
class ClassificationResult(BaseModel):
    index: int
    category_type: str = Field(description="분류: 'MEAL', 'SUPPLIES', 'TRAFFIC', 'FEE', 'OTHER'")
    account_code: str = Field(description="계정과목코드")
    account_name: str = Field(description="계정과목명")
    remark: str = Field(description="전표 적요")

class BatchClassification(BaseModel):
    items: list[ClassificationResult]

def detect_card_info(df_raw):
    """카드사 종류 및 진짜 헤더 행 위치 자동 감지"""
    for r_idx in range(min(15, len(df_raw))):
        row_str = " ".join([str(x) for x in df_raw.iloc[r_idx].values if pd.notna(x)]).replace("\n", " ")
        if "이용가맹점(은행)명" in row_str or "이용카드" in row_str:
            return "우리카드", r_idx
        elif "사업자번호" in row_str and "가맹점명" in row_str and "승인번호" in row_str:
            return "신한카드", r_idx
        elif "고객명" in row_str and "가맹점명" in row_str and "이용원금" in row_str:
            return "하나카드", r_idx
        elif any(k in row_str for k in ["가맹점명", "가맹점", "상호"]) and any(k in row_str for k in ["이용일", "일자"]):
            return "법인카드사", r_idx
    return "법인카드사", 0

def get_col_idx(cols, exact_matches, partial_matches, fallback=0):
    for match in exact_matches:
        for i, c in enumerate(cols):
            if match == c: return i
    for match in partial_matches:
        for i, c in enumerate(cols):
            if match in c: return i
    return fallback

def clean_amount(val):
    if pd.isna(val): return 0
    if isinstance(val, (int, float)): return int(val)
    val_str = str(val).replace(",", "").strip()
    match = re.search(r"-?\d+", val_str)
    return int(match.group()) if match else 0

def clean_card(val):
    if pd.isna(val): return ""
    v = str(val).strip().replace("-", "").replace("*", "")
    v = re.sub(r'\.0$', '', v)
    return v

def clean_date(val, base_year):
    if pd.isna(val): return ""
    val_str = str(val).strip()
    numbers = re.findall(r"\d+", val_str)
    if len(numbers) >= 3:
        y, m, d = numbers[0], numbers[1].zfill(2), numbers[2].zfill(2)
        if len(y) == 2: y = "20" + y
        return f"{y}{m}{d}"
    elif len(numbers) == 2:
        m, d = numbers[0].zfill(2), numbers[1].zfill(2)
        return f"{base_year}{m}{d}"
    return val_str[:8]

# ------------------------------------------------------------------------------
# AI 모델 호출 함수
# ------------------------------------------------------------------------------
def classify_with_gemini(client, batch_df):
    system_instruction = """
    당신은 대한민국 기업의 전문 회계 에이전트입니다.
    제공된 법인카드 승인내역을 사내 7자리 계정과목 기준에 따라 분류하세요.

    [핵심 사내 회계 기준 (7자리)]
    1. 식대/카페/음식점/회식: 기본 8110000(복리후생비). 
       단, 카드번호 끝자리 '4015', '8348'은 무조건 8130003(접대비(신용카드))
    2. 소모품/사무용품/잡화: 끝자리 '5630', '1072', '4760'은 8300000(소모품비-판관), 그 외는 5300000(소모품비-제조)
    3. 여비교통/주유/차량: 8120000(여비교통비) 또는 8220000(차량유지비)
    4. 수수료/IT/통신/구독료/알림메시지: 8310000(지급수수료) 또는 8140000(통신비)
    5. 보험료: 8210000(보험료), 협회비: 8490000(협회비), 수출비: 8380000(수출제비용)

    * 적요는 가맹점명 없이 사용 목적만 간결하게 작성하세요. (예: 야간식대, 차량주유, 사무용품 등)
    """

    data_summary = []
    for idx, row in batch_df.iterrows():
        data_summary.append({
            "index": int(idx),
            "merchant": str(row.get("가맹점명", "")).strip(),
            "amount": row.get("이용금액", 0),
            "card_last4": str(row.get("카드번호", ""))[-4:],
        })

    prompt = f"다음 결제 내역을 사내 7자리 규칙에 맞게 분류해줘:\n{json.dumps(data_summary, ensure_ascii=False)}"

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=BatchClassification,
        ),
    )
    return json.loads(response.text)

# -------------------------------------------------------------
# 파일 업로드 UI & 데이터 정제
# -------------------------------------------------------------
uploaded_file = st.file_uploader("신한/우리/하나카드 이용내역 엑셀 파일(.xlsx, .xls, .csv) 업로드", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        is_csv = uploaded_file.name.lower().endswith(".csv")
        raw_df = pd.read_csv(uploaded_file, header=None) if is_csv else pd.read_excel(uploaded_file, header=None)
        
        # 카드사 및 헤더 행 자동 감지
        detected_card, header_row_idx = detect_card_info(raw_df)
        
        uploaded_file.seek(0)
        df_card = pd.read_csv(uploaded_file, header=header_row_idx) if is_csv else pd.read_excel(uploaded_file, header=header_row_idx)
    except Exception as e:
        uploaded_file.seek(0)
        df_card = pd.read_csv(uploaded_file) if is_csv else pd.read_excel(uploaded_file)
        detected_card = "법인카드사"

    df_card.columns = [str(c).replace('\n', '').replace(' ', '').strip() for c in df_card.columns]
    cols = df_card.columns.tolist()

    st.success(f"💳 감지된 카드사: **{detected_card}** (헤더 자동 매핑 완료)")
    st.dataframe(df_card.head())

    # 3개 카드사 전용 열 자동 매핑
    date_idx = get_col_idx(cols, ["이용일", "이용일자", "승인일자", "일자"], ["일자", "이용일", "일시"], 0)
    card_idx = get_col_idx(cols, ["이용카드", "카드번호", "카드"], ["카드"], 1 if len(cols) > 1 else 0)
    mer_idx = get_col_idx(cols, ["이용가맹점(은행)명", "가맹점명", "이용가맹점명", "상호"], ["가맹점", "상호"], 2 if len(cols) > 2 else 0)
    amt_idx = get_col_idx(cols, ["이용원금", "이용금액(현지금액)", "이용금액", "승인금액"], ["금액", "원금", "승인", "결제"], 3 if len(cols) > 3 else 0)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: sel_date = st.selectbox("일자 열", cols, index=date_idx)
    with col2: sel_card = st.selectbox("카드번호 열", cols, index=card_idx)
    with col3: sel_merchant = st.selectbox("가맹점명 열", cols, index=mer_idx)
    with col4: sel_amount = st.selectbox("이용금액 열", cols, index=amt_idx)
    with col5: sel_card_corp = st.text_input("대변 거래처명", value=detected_card)

    if st.button("🚀 아마란스 10 양식 전표 생성 (주말 색칠 포함)"):
        client = genai.Client(api_key=GEMINI_API_KEY)

        with st.spinner("데이터 정제 및 Gemini 3.6 Flash 분개 생성 중..."):
            temp_df = pd.DataFrame()
            temp_df["이용일자"] = df_card[sel_date].apply(lambda x: clean_date(x, default_year))
            temp_df["카드번호"] = df_card[sel_card].apply(clean_card)
            temp_df["가맹점명"] = df_card[sel_merchant].astype(str)
            temp_df["이용금액"] = df_card[sel_amount].apply(clean_amount)

            # 불필요한 하단 합계, 여백 줄, 잔여 헤더 행 완벽 필터링
            temp_df = temp_df[
                (temp_df["가맹점명"].str.strip() != "") & 
                (temp_df["가맹점명"] != "nan") & 
                (temp_df["가맹점명"] != "None") & 
                (~temp_df["가맹점명"].str.contains("이용가맹점|가맹점명|이하여백", na=False)) &
                (temp_df["이용금액"] != 0)
            ].reset_index(drop=True)

            if len(temp_df) == 0:
                st.warning("유효한 결제 내역을 찾지 못했습니다. 드롭다운 열을 확인해주세요.")
                st.stop()

            # AI 분석 실행
            ai_result = classify_with_gemini(client, temp_df)
            result_map = {item["index"]: item for item in ai_result.get("items", [])}

            ADMIN_CARDS = {"5630", "1072", "4760"}
            ENTERTAIN_CARDS = {"4015", "8348"}
            
            today_str = datetime.now().strftime("%Y%m%d")
            line_seq = 1

            rows = []
            weekend_flags = []

            for idx, row in temp_df.iterrows():
                trans_date = row["이용일자"]
                raw_merchant = row["가맹점명"].strip()
                total_amt = row["이용금액"]
                last4 = row["카드번호"][-4:] if len(row["카드번호"]) >= 4 else row["카드번호"]

                # 주소 제거: 2개 이상의 연속 공백 뒤 주소 잘라내기
                clean_merchant_name = re.split(r'\s{2,}', raw_merchant)[0].strip()

                try:
                    dt = datetime.strptime(trans_date, "%Y%m%d")
                    is_weekend = dt.weekday() >= 5
                except:
                    is_weekend = False

                ai_info = result_map.get(idx, {
                    "category_type": "OTHER",
                    "account_code": "5300000",
                    "account_name": "소모품비(제조)",
                    "remark": "카드대금",
                })

                acct_code = ai_info.get("account_code", "5300000")
                remark = ai_info.get("remark", "카드대금")
                cat_type = ai_info.get("category_type", "")

                # 하드 룰 적용
                if last4 in ENTERTAIN_CARDS and ("식" in raw_merchant or "카페" in raw_merchant or cat_type == "MEAL" or "8110000" in acct_code):
                    acct_code = "8130003"
                    remark = "거래처 접대"
                elif "소모품" in ai_info.get("account_name", "") or cat_type == "SUPPLIES":
                    if last4 in ADMIN_CARDS:
                        acct_code = "8300000"
                    else:
                        acct_code = "5300000"

                # 적요 포맷: [YYMMDD] 가맹점명 적요 (카드:XXXX)
                usage_yymmdd = trans_date[2:8] if len(trans_date) == 8 else trans_date
                final_remark = f"[{usage_yymmdd}] {clean_merchant_name} {remark} (카드:{last4})"

                # 1) 차변 (차대구분 3)
                rows.append({
                    "회계단위": "1000",
                    "작성일자": today_str,
                    "작성번호": "100",
                    "라인순번": line_seq,
                    "전표유형": "1",
                    "차대구분": "3",
                    "계정과목": acct_code,
                    "거래처명": clean_merchant_name,
                    "계정금액": total_amt,
                    "적요": final_remark,
                })
                weekend_flags.append(is_weekend)
                line_seq += 1

                # 2) 대변 (차대구분 4)
                rows.append({
                    "회계단위": "1000",
                    "작성일자": today_str,
                    "작성번호": "100",
                    "라인순번": line_seq,
                    "전표유형": "1",
                    "차대구분": "4",
                    "계정과목": default_credit_account,
                    "거래처명": sel_card_corp,
                    "계정금액": total_amt,
                    "적요": final_remark,
                })
                weekend_flags.append(is_weekend)
                line_seq += 1

            df_amaranth = pd.DataFrame(rows)
            st.success("✅ 아마란스 10 양식 변환이 완료되었습니다!")
            st.dataframe(df_amaranth, use_container_width=True)

            # 엑셀 서식 적용 (주말 결제 노란색 하이라이트)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_amaranth.to_excel(writer, index=False, sheet_name="아마란스전표업로드")
                worksheet = writer.sheets["아마란스전표업로드"]
                
                weekend_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
                
                for i, is_week in enumerate(weekend_flags):
                    if is_week:
                        for cell in worksheet[i + 2]:
                            cell.fill = weekend_fill

            excel_data = output.getvalue()

            st.download_button(
                label="📥 아마란스 10 업로드용 엑셀 다운로드",
                data=excel_data,
                file_name=f"아마란스10_{sel_card_corp}_전표_{today_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )