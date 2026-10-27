# ============================================================
# 반도체 웨이퍼 수율 시뮬레이터
# 실행 방법: streamlit run main.py
# ============================================================
# [용어 설명]
# - 웨이퍼(wafer): 반도체 칩을 만드는 둥근 실리콘 판
# - 다이(die): 웨이퍼 위에 찍힌 칩 하나 (이 프로그램에서는 '칩'이라고 부름)
# - 결함(defect): 먼지, 흠집처럼 칩을 망가뜨리는 문제
# - 수율(yield): 전체 칩 중 정상 칩(양품)의 비율
# - 결함 밀도(D0): 1cm² 안에 평균 몇 개의 결함이 있는지
# ============================================================

# ---------- 1. 라이브러리 불러오기 ----------
import streamlit as st                 # 웹 화면을 만들어 주는 라이브러리
import numpy as np                     # 숫자·난수 계산용
import pandas as pd                    # 표(데이터프레임)용
import plotly.graph_objects as go      # 그래프(웨이퍼 맵 등)를 그리는 라이브러리

# ---------- 2. 기본 화면 설정 ----------
st.set_page_config(page_title="웨이퍼 수율 시뮬레이터", page_icon="🔬", layout="wide")

MAX_DICE = 30000   # 칩이 너무 많으면 프로그램이 느려지므로 최대 개수를 정해 둔다


# ---------- 3. 핵심 기능 함수들 ----------
def validate_inputs(wafer_d, die_w, die_h, edge_excl):
    """입력값이 올바른지 검사한다. 문제가 있으면 안내 문장을, 없으면 None을 돌려준다."""
    usable_r = wafer_d / 2 - edge_excl          # 칩을 찍을 수 있는 반지름
    if usable_r <= 0:
        return "가장자리 제외 폭이 너무 커서 칩을 찍을 공간이 없어요."
    if die_w > 2 * usable_r or die_h > 2 * usable_r:
        return "칩이 웨이퍼보다 커요. 칩 크기를 줄이거나 웨이퍼를 키워 주세요."
    # 원의 넓이 / 칩 넓이 = 대략적인 칩 개수
    estimated = np.pi * usable_r ** 2 / (die_w * die_h)
    if estimated > MAX_DICE:
        return f"칩이 너무 작아서 {MAX_DICE:,}개를 넘어요. 칩 크기를 키워 주세요."
    return None


def make_die_grid(wafer_d, die_w, die_h, edge_excl):
    """웨이퍼 안에 들어가는 칩들의 위치를 계산해서 표(DataFrame)로 돌려준다.
    웨이퍼의 중심이 (0, 0)이고, 칩은 중심을 기준으로 격자 모양으로 놓는다.
    칩의 네 모서리가 모두 '사용 가능한 원' 안에 있어야 한다."""
    usable_r = wafer_d / 2 - edge_excl

    # 격자 번호 범위 (원을 충분히 덮을 만큼 넉넉하게 잡는다)
    n_x = int(np.ceil(usable_r / die_w)) + 1
    n_y = int(np.ceil(usable_r / die_h)) + 1
    ix = np.arange(-n_x, n_x + 1)
    iy = np.arange(-n_y, n_y + 1)
    grid_x, grid_y = np.meshgrid(ix, iy)         # 모든 (가로번호, 세로번호) 조합 만들기
    grid_x = grid_x.ravel()                      # 1차원으로 펴기
    grid_y = grid_y.ravel()

    # 각 칩의 왼쪽/오른쪽/아래/위 좌표 (칩 하나의 중심이 (0,0) 근처에 오도록 반 칸 이동)
    x0 = grid_x * die_w - die_w / 2
    x1 = x0 + die_w
    y0 = grid_y * die_h - die_h / 2
    y1 = y0 + die_h

    # 원점에서 가장 먼 모서리까지의 거리가 사용 가능 반지름 이하이면 '원 안에 있는 칩'
    far_x = np.maximum(np.abs(x0), np.abs(x1))
    far_y = np.maximum(np.abs(y0), np.abs(y1))
    inside = far_x ** 2 + far_y ** 2 <= usable_r ** 2

    dies = pd.DataFrame({
        "ix": grid_x[inside], "iy": grid_y[inside],
        "x0": x0[inside], "x1": x1[inside],
        "y0": y0[inside], "y1": y1[inside],
    })
    # 칩마다 고유 번호표(key)를 붙인다. 나중에 결함이 어느 칩에 떨어졌는지 찾을 때 사용.
    dies["key"] = dies["ix"] * 100000 + dies["iy"]
    return dies


def throw_defects(rng, wafer_d, d0):
    """웨이퍼 위에 결함을 무작위로 뿌린다. 결함의 x, y 좌표를 돌려준다.
    - 결함 개수: 평균 (결함밀도 x 웨이퍼 넓이)인 포아송 분포에서 뽑는다.
    - 결함 위치: 웨이퍼 원 안에서 균일하게(고르게) 뽑는다."""
    radius = wafer_d / 2                              # mm
    area_cm2 = np.pi * (radius / 10) ** 2             # 웨이퍼 넓이 (cm²), 10mm = 1cm
    n_defects = rng.poisson(d0 * area_cm2)            # 결함 개수

    # 원 안에서 고르게 뽑으려면 반지름에 제곱근(sqrt)을 써야 한다
    r = radius * np.sqrt(rng.random(n_defects))
    theta = rng.random(n_defects) * 2 * np.pi         # 각도 (0 ~ 360도)
    return r * np.cos(theta), r * np.sin(theta)


def find_bad_dice(dies, dx, dy, die_w, die_h):
    """결함이 떨어진 칩을 찾아서 '불량이면 True'인 배열을 돌려준다."""
    if len(dx) == 0:                                  # 결함이 하나도 없으면 전부 양품
        return np.zeros(len(dies), dtype=bool)
    # 결함 좌표가 몇 번째 칩 칸에 속하는지 계산 (칸 번호 = 좌표 / 칩 크기, 내림)
    ix_def = np.floor((dx + die_w / 2) / die_w).astype(np.int64)
    iy_def = np.floor((dy + die_h / 2) / die_h).astype(np.int64)
    key_def = ix_def * 100000 + iy_def
    # 칩의 번호표가 결함의 번호표 목록에 있으면 불량
    return np.isin(dies["key"].to_numpy(), key_def)


def run_monte_carlo(rng, dies, wafer_d, die_w, die_h, d0, n_runs):
    """웨이퍼 한 장 시뮬레이션을 n_runs번 반복해서 수율 목록(0~1)을 돌려준다."""
    yields = []
    for _ in range(n_runs):
        dx, dy = throw_defects(rng, wafer_d, d0)
        bad = find_bad_dice(dies, dx, dy, die_w, die_h)
        yields.append(1 - bad.mean())                 # 수율 = 1 - 불량 비율
    return np.array(yields)


def theory_poisson(die_area_cm2, d0):
    """포아송 모델 이론 수율: Y = exp(-A x D0)  (결함이 완전히 무작위일 때)"""
    return float(np.exp(-die_area_cm2 * d0))


def theory_murphy(die_area_cm2, d0):
    """머피 모델 이론 수율: Y = ((1 - exp(-AD)) / AD)^2  (결함이 칩마다 다르게 퍼질 때)"""
    ad = die_area_cm2 * d0
    if ad == 0:                                       # 0으로 나누기 방지
        return 1.0
    return float(((1 - np.exp(-ad)) / ad) ** 2)


def draw_wafer_map(wafer_d, edge_excl, dies, is_bad, dx, dy):
    """웨이퍼 맵 그림을 만든다. 초록=양품 칩, 빨강=불량 칩, X 표시=결함 위치"""
    radius = wafer_d / 2
    usable_r = radius - edge_excl
    fig = go.Figure()

    # 웨이퍼 바깥 원 + 가장자리 제외 경계선(점선)
    shapes = [
        dict(type="circle", x0=-radius, y0=-radius, x1=radius, y1=radius,
             line=dict(color="gray", width=2), fillcolor="rgba(200,200,200,0.25)"),
        dict(type="circle", x0=-usable_r, y0=-usable_r, x1=usable_r, y1=usable_r,
             line=dict(color="gray", width=1, dash="dash")),
    ]

    if len(dies) <= 4000:
        # 칩이 적당히 많으면 사각형으로 하나씩 그린다
        for x0, y0, x1, y1, bad in zip(dies["x0"], dies["y0"], dies["x1"], dies["y1"], is_bad):
            color = "rgba(229,57,53,0.75)" if bad else "rgba(76,175,80,0.75)"
            shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                               fillcolor=color, line=dict(color="white", width=0.5)))
    else:
        # 칩이 너무 많으면 사각형 대신 점으로 그려서 속도를 지킨다
        cx = (dies["x0"] + dies["x1"]) / 2
        cy = (dies["y0"] + dies["y1"]) / 2
        colors = np.where(is_bad, "rgba(229,57,53,0.75)", "rgba(76,175,80,0.75)")
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers",
                                 marker=dict(size=3, color=colors), hoverinfo="skip"))

    # 결함 위치를 X로 표시 (너무 많으면 3000개까지만)
    if len(dx) > 0:
        fig.add_trace(go.Scatter(x=dx[:3000], y=dy[:3000], mode="markers",
                                 marker=dict(size=5, color="black", symbol="x"),
                                 name="결함"))

    fig.update_layout(shapes=shapes, height=650, showlegend=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(range=[-radius * 1.05, radius * 1.05], title="x (mm)")
    fig.update_yaxes(range=[-radius * 1.05, radius * 1.05], title="y (mm)",
                     scaleanchor="x", scaleratio=1)   # 가로세로 비율 1:1 (원이 찌그러지지 않게)
    return fig


# ---------- 4. 화면 제목 ----------
st.title("🔬 반도체 웨이퍼 수율 시뮬레이터")
st.caption("웨이퍼 크기·칩 크기·결함 밀도를 바꿔 가며 수율이 어떻게 달라지는지 확인해 보세요.")

with st.expander("📖 이 프로그램은 어떻게 동작하나요?"):
    st.markdown(
        "1. 원형 웨이퍼 안에 들어가는 칩의 위치를 격자로 계산해요.\n"
        "2. 결함 밀도를 바탕으로 결함 개수를 무작위로 뽑아 웨이퍼 위에 뿌려요.\n"
        "3. 결함이 떨어진 칩은 **불량(빨강)**, 아닌 칩은 **양품(초록)**이에요.\n"
        "4. 이를 여러 번 반복(몬테카를로 시뮬레이션)해서 이론 수율과 비교해요.\n\n"
        "- **포아송 모델**: 결함이 완전히 무작위로 퍼질 때의 이론식 `Y = e^(-A·D0)`\n"
        "- **머피 모델**: 결함이 웨이퍼마다 불균일할 때를 고려한 이론식"
    )

# ---------- 5. 사이드바: 입력값 ----------
with st.sidebar:
    st.header("⚙️ 입력값")
    wafer_d = st.selectbox("웨이퍼 지름 (mm)", [100, 150, 200, 300], index=3)
    die_w = st.number_input("칩 가로 길이 (mm)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
    die_h = st.number_input("칩 세로 길이 (mm)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
    edge_excl = st.number_input("가장자리 제외 폭 (mm)", min_value=0.0, max_value=20.0, value=3.0, step=0.5,
                                help="웨이퍼 가장자리는 품질이 나빠서 칩을 찍지 않는 구간이에요.")
    d0 = st.number_input("결함 밀도 D0 (개/cm²)", min_value=0.0, max_value=5.0, value=0.3, step=0.05,
                         format="%.2f")
    wafer_cost = st.number_input("웨이퍼 1장 가격 (달러)", min_value=0.0, max_value=100000.0,
                                 value=5000.0, step=100.0)
    seed = st.number_input("난수 시드 (바꾸면 다른 결과)", min_value=0, max_value=99999, value=42, step=1)

# ---------- 6. 입력 검사 및 칩 격자 만들기 ----------
try:
    error = validate_inputs(wafer_d, die_w, die_h, edge_excl)
    if error:
        st.error(error)              # 입력이 잘못되면 안내 문구를 보여 주고
        st.stop()                    # 아래 계산은 하지 않고 멈춘다

    dies = make_die_grid(float(wafer_d), die_w, die_h, edge_excl)
    total = len(dies)                # 웨이퍼 1장에 들어가는 전체 칩 수
    if total == 0:
        st.error("웨이퍼에 들어가는 칩이 없어요. 칩 크기나 가장자리 제외 폭을 조절해 주세요.")
        st.stop()

    die_area_cm2 = die_w * die_h / 100               # mm² -> cm² (100mm² = 1cm²)
    y_poisson = theory_poisson(die_area_cm2, d0)
    y_murphy = theory_murphy(die_area_cm2, d0)
except Exception as e:               # 예상 못한 오류가 나도 프로그램이 멈추지 않게 한다
    st.error(f"입력값을 처리하는 중 오류가 발생했어요: {e}")
    st.stop()

# ---------- 7. 탭 만들기 ----------
tab1, tab2, tab3 = st.tabs(["🗺️ 웨이퍼 맵", "🎲 반복 시뮬레이션", "📈 결함 밀도별 비교"])

# ======================================================
# 탭 1: 웨이퍼 한 장 시뮬레이션
# ======================================================
with tab1:
    try:
        rng = np.random.default_rng(int(seed))       # 시드가 같으면 같은 결과가 나온다
        dx, dy = throw_defects(rng, float(wafer_d), d0)
        is_bad = find_bad_dice(dies, dx, dy, die_w, die_h)

        bad_cnt = int(is_bad.sum())
        good_cnt = total - bad_cnt
        y_sim = good_cnt / total * 100               # 이번 웨이퍼의 수율(%)

        # 칩 1개 원가 = 웨이퍼 가격 / 양품 칩 수 (양품이 0개면 계산 불가)
        cost_text = f"${wafer_cost / good_cnt:,.2f}" if good_cnt > 0 else "계산 불가"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("웨이퍼당 전체 칩", f"{total:,}개")
        c2.metric("양품 칩", f"{good_cnt:,}개")
        c3.metric("불량 칩", f"{bad_cnt:,}개")
        c4.metric("이번 웨이퍼 수율", f"{y_sim:.1f}%",
                  f"이론(포아송) 대비 {y_sim - y_poisson * 100:+.1f}%p")
        c5.metric("양품 칩 1개 원가", cost_text)

        st.plotly_chart(draw_wafer_map(float(wafer_d), edge_excl, dies, is_bad, dx, dy),
                        use_container_width=True)
        st.caption(f"결함 {len(dx):,}개가 떨어졌어요. 같은 입력이어도 시드를 바꾸면 결과가 달라져요.")
    except Exception as e:
        st.error(f"웨이퍼 맵을 만드는 중 오류가 발생했어요: {e}")

# ======================================================
# 탭 2: 몬테카를로 (여러 번 반복)
# ======================================================
with tab2:
    try:
        n_runs = st.slider("반복 횟수 (웨이퍼 몇 장을 시뮬레이션할까요?)", 10, 500, 100, step=10)
        with st.spinner("시뮬레이션 중..."):
            rng2 = np.random.default_rng(int(seed))
            yields = run_monte_carlo(rng2, dies, float(wafer_d), die_w, die_h, d0, n_runs) * 100

        # 통계 표
        stat_df = pd.DataFrame({
            "항목": ["시뮬레이션 평균", "표준편차", "최솟값", "최댓값", "이론값(포아송)", "이론값(머피)"],
            "수율(%)": [yields.mean(), yields.std(), yields.min(), yields.max(),
                        y_poisson * 100, y_murphy * 100],
        })
        stat_df["수율(%)"] = stat_df["수율(%)"].round(2)

        left, right = st.columns([1, 2])
        left.dataframe(stat_df, use_container_width=True, hide_index=True)
        left.metric("평균과 포아송 이론값의 차이", f"{yields.mean() - y_poisson * 100:+.2f}%p")

        # 히스토그램: 수율이 어떻게 퍼지는지 보여 준다
        fig_hist = go.Figure(go.Histogram(x=yields, nbinsx=20, name="시뮬레이션"))
        fig_hist.add_vline(x=y_poisson * 100, line_dash="dash", line_color="red",
                           annotation_text="포아송 이론값")
        fig_hist.update_layout(title=f"웨이퍼 {n_runs}장의 수율 분포",
                               xaxis_title="수율 (%)", yaxis_title="웨이퍼 수", height=420)
        right.plotly_chart(fig_hist, use_container_width=True)
        st.caption("웨이퍼마다 결함 개수가 달라서 수율도 조금씩 달라져요. 반복할수록 평균이 이론값에 가까워져요.")
    except Exception as e:
        st.error(f"반복 시뮬레이션 중 오류가 발생했어요: {e}")

# ======================================================
# 탭 3: 결함 밀도를 바꿔 가며 수율 비교
# ======================================================
with tab3:
    try:
        sweep_runs = st.slider("결함 밀도마다 반복할 횟수", 10, 100, 30, step=10, key="sweep_runs")
        d0_list = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]   # 비교할 결함 밀도 목록

        rng3 = np.random.default_rng(int(seed))
        rows = []
        with st.spinner("결함 밀도별로 계산 중..."):
            for d in d0_list:
                sim_y = run_monte_carlo(rng3, dies, float(wafer_d), die_w, die_h, d, sweep_runs).mean() * 100
                rows.append({
                    "결함 밀도(개/cm²)": d,
                    "시뮬레이션 수율(%)": round(sim_y, 2),
                    "포아송 이론(%)": round(theory_poisson(die_area_cm2, d) * 100, 2),
                    "머피 이론(%)": round(theory_murphy(die_area_cm2, d) * 100, 2),
                })
        sweep_df = pd.DataFrame(rows)

        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_df["결함 밀도(개/cm²)"], y=sweep_df["포아송 이론(%)"],
                                       mode="lines", name="포아송 이론"))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["결함 밀도(개/cm²)"], y=sweep_df["머피 이론(%)"],
                                       mode="lines", name="머피 이론"))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["결함 밀도(개/cm²)"], y=sweep_df["시뮬레이션 수율(%)"],
                                       mode="markers", name="시뮬레이션 평균",
                                       marker=dict(size=10, symbol="diamond")))
        fig_sweep.update_layout(title=f"결함 밀도에 따른 수율 (칩 크기 {die_w}×{die_h}mm)",
                                xaxis_title="결함 밀도 (개/cm²)", yaxis_title="수율 (%)", height=450)
        st.plotly_chart(fig_sweep, use_container_width=True)
        st.dataframe(sweep_df, use_container_width=True, hide_index=True)

        csv = sweep_df.to_csv(index=False).encode("utf-8-sig")   # utf-8-sig: 엑셀에서 한글 안 깨짐
        st.download_button("📥 결과 CSV 다운로드", csv, "yield_sweep.csv", "text/csv")
    except Exception as e:
        st.error(f"결함 밀도별 비교 중 오류가 발생했어요: {e}")
