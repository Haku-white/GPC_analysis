import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
import re
import numpy as np

st.set_page_config(page_title="GPC Data Visualization", layout="wide")

st.markdown("""
<style>
    div[data-testid="stFileUploader"] section {
        padding: 5rem 2rem;
        min-height: 250px;
        border-width: 2px;
    }

    div[data-testid="stDataFrame"] {
        color: black !important;
    }
    div[data-testid="stDataFrame"] table {
        color: black !important;
    }
    div[data-testid="stDataFrame"] th {
        color: black !important;
    }
    div[data-testid="stDataFrame"] td {
        color: black !important;
    }
    div[data-testid="stDataFrame"] [data-testid="styled-table"] {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("#### ゲル浸透クロマトグラフィー(GPC) データ可視化・比較ツール")
st.caption("複数のGPCデータファイルの波形を重ねて描画し、分子量データの表表示、及びグラフのカスタマイズが可能です。")
st.divider()

if 'expanded_tables' not in st.session_state:
    st.session_state.expanded_tables = {}

def get_series_name(filename):
    match = re.search(r"(_\d{8})\.txt$", filename, re.IGNORECASE)
    if match:
        return filename[:match.start()]
    elif filename.lower().endswith(".txt"):
        return filename[:-4]
    return filename

@st.cache_data(show_spinner=False)
def parse_gpc_file(file_content, filename):
    lines = file_content.decode("utf-8", errors="replace").splitlines()

    df_chrom = pd.DataFrame()
    df_mw = pd.DataFrame()

    idx_mw = -1
    idx_chrom = -1

    for i, line in enumerate(lines):
        if "[Average Molecular Weight Table(AD2)]" in line:
            idx_mw = i
        if "[LC Chromatogram(AD2)]" in line:
            idx_chrom = i

    if idx_mw != -1:
        header_idx = -1
        for i in range(idx_mw, min(idx_mw + 10, len(lines))):
            if "Mn" in lines[i] or "Mw" in lines[i]:
                header_idx = i
                break

        if header_idx != -1:
            table_lines = [lines[header_idx]]
            for i in range(header_idx + 2, min(header_idx + 50, len(lines))):
                if not lines[i].strip():
                    break
                table_lines.append(lines[i])

            table_str = "\n".join(table_lines)
            try:
                temp_df = pd.read_csv(io.StringIO(table_str), sep='\t', engine='python')
                if len(temp_df.columns) <= 1:
                    temp_df = pd.read_csv(io.StringIO(table_str), sep=',', engine='python')

                req_cols = []
                rename_map = {}
                for col in temp_df.columns:
                    col_str = str(col).strip()
                    if col_str in ["Mn", "Mw", "%"]:
                        req_cols.append(col)
                        rename_map[col] = col_str
                    elif col_str == "Mw/Mn":
                        req_cols.append(col)
                        rename_map[col] = "Ð"

                if req_cols:
                    df_mw = temp_df[req_cols].rename(columns=rename_map)
                    for col in df_mw.columns:
                        if col in ["Mn", "Mw", "Ð", "%"]:
                            df_mw[col] = pd.to_numeric(df_mw[col], errors='coerce')
                    df_mw = df_mw.dropna(how='all').reset_index(drop=True)
            except Exception as e:
                pass

    # Parse Chromatogram
    if idx_chrom != -1:
        start_idx = idx_chrom + 8
        if start_idx < len(lines):
            chrom_lines = []
            for i in range(start_idx, len(lines)):
                line = lines[i].strip()
                if not line: # 空行があったら終了
                    break
                chrom_lines.append(line)

            if chrom_lines:
                data_str = "\n".join(chrom_lines)
                try:
                    df_c = pd.read_csv(io.StringIO(data_str), sep='\t', header=None, engine='python')
                    if len(df_c.columns) <= 1:
                        df_c = pd.read_csv(io.StringIO(data_str), sep=',', header=None, engine='python')

                    if len(df_c.columns) >= 2:
                        df_c = df_c.iloc[:, [0, 1]]
                        df_c.columns = ["Retention time [min]", "Intensity"]
                        df_c["Retention time [min]"] = pd.to_numeric(df_c["Retention time [min]"], errors='coerce')
                        df_c["Intensity"] = pd.to_numeric(df_c["Intensity"], errors='coerce')
                        df_chrom = df_c.dropna()
                except Exception as e:
                    pass

    return df_chrom, df_mw


@st.cache_data(show_spinner=False)
def normalize_data(df, x_min, x_max):
    # Retrieve data within visible limits using strictly valid numeric boundaries
    mask = (df["Retention time [min]"] >= x_min) & (df["Retention time [min]"] <= x_max)
    sub_df = df[mask]

    if not sub_df.empty:
        min_val = sub_df["Intensity"].min()
        max_val = sub_df["Intensity"].max()
        val_range = max_val - min_val

        # Avoid division by 0 if range is somehow 0
        if val_range != 0:
            df["Intensity_Norm"] = (df["Intensity"] - min_val) / val_range
        else:
            df["Intensity_Norm"] = df["Intensity"] - min_val
    else:
        # Failsafe if data is totally outside limits
        df["Intensity_Norm"] = df["Intensity"] - df["Intensity"].min()
        val_range = df["Intensity"].max() - df["Intensity"].min()
        if val_range != 0:
            df["Intensity_Norm"] = df["Intensity_Norm"] / val_range

    return df


import pickle

# --- Main App ---
col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_files = st.file_uploader("GPCデータファイル (.txt) をドラッグ＆ドロップ", type=["txt"], accept_multiple_files=True)
with col_up2:
    project_file = st.file_uploader("💾 プロジェクトファイルの復元 (.gpc)", type=["gpc"])

if not uploaded_files and project_file is None:
    st.markdown("""
    <style>
        div[data-testid="stFileUploader"] section {
            padding: 5rem 2rem;
            min-height: 250px;
            border-width: 2px;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        div[data-testid="stFileUploader"] section {
            padding: 1rem 1rem;
            min-height: 80px;
        }
    </style>
    """, unsafe_allow_html=True)

data_dict = {}

if project_file is not None:
    try:
        project_data = pickle.loads(project_file.getvalue())
        data_dict = project_data.get('data_dict', {})
        loaded_state = project_data.get('session_state', {})

        # １回だけセッションステートを復元する
        if st.session_state.get('last_loaded_project') != project_file.name:
            # 現在のセッションをクリア（必要なもの以外）して上書き
            st.session_state.clear()
            for k, v in loaded_state.items():
                st.session_state[k] = v
            st.session_state['last_loaded_project'] = project_file.name
            st.rerun()
    except Exception as e:
        st.error(f"プロジェクトの読み込みに失敗しました: {e}")
elif uploaded_files:
    for file in uploaded_files:
        try:
            file_bytes = file.getvalue()
            df_chrom, df_mw = parse_gpc_file(file_bytes, file.name)

            if not df_chrom.empty:
                series_name = get_series_name(file.name)
                data_dict[file.name] = {
                    'series_name': series_name,
                    'df_chrom': df_chrom,
                    'df_mw': df_mw
                }
            else:
                st.warning(f"{file.name}: クロマトグラムデータが見つかりませんでした。[LC Chromatogram(AD2)] の8行下以降のデータをご確認ください。")
        except Exception as e:
            st.error(f"{file.name} の読み込みに失敗しました: {e}")

else:
    st.info("💡 サイドバーやメイン画面からデータファイル (.txt) または保存したプロジェクト (.gpc) をアップロードしてください。")

if True:
    if data_dict:
        # --- Sidebar settings ---
        if st.sidebar.button("🔄 設定を初期状態に戻す"):
            # Clear all session state variables to reset the app
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.sidebar.divider()
        st.sidebar.header("📑 凡例・グラフの表示順")

        file_names = list(data_dict.keys())
        # 新規ファイル追加等の変化があれば順番リストを更新
        if 'ordered_files' not in st.session_state or set(st.session_state.ordered_files) != set(file_names):
            st.session_state.ordered_files = file_names

        st.sidebar.markdown("ファイルの順番を「▲」「▼」で入れ替え、✔で表示/非表示を切り替えられます。")

        ordered_files = []
        for i, filename in enumerate(st.session_state.ordered_files):
            col1, col2, col3 = st.sidebar.columns([6, 2, 2])
            with col1:
                is_checked = st.checkbox(data_dict[filename]['series_name'], value=st.session_state.get(f"vis_{filename}", True), key=f"vis_chk_{filename}")
                st.session_state[f"vis_{filename}"] = is_checked
                if is_checked:
                    ordered_files.append(filename)
            with col2:
                if i > 0:
                    if st.button("▲", key=f"up_{filename}"):
                        st.session_state.ordered_files[i-1], st.session_state.ordered_files[i] = st.session_state.ordered_files[i], st.session_state.ordered_files[i-1]
                        st.rerun()
            with col3:
                if i < len(st.session_state.ordered_files) - 1:
                    if st.button("▼", key=f"down_{filename}"):
                        st.session_state.ordered_files[i+1], st.session_state.ordered_files[i] = st.session_state.ordered_files[i], st.session_state.ordered_files[i+1]
                        st.rerun()

        # --- Display MW Tables ---
        st.subheader("📊 分子量データ (Average Molecular Weight Table)")
        for filename in ordered_files:
            data = data_dict[filename]
            df_mw = data['df_mw']
            if not df_mw.empty:
                col1, col2 = st.columns([8, 2])

                with col1:
                    st.markdown(f"**{data['series_name']}**")

                if filename not in st.session_state.expanded_tables:
                    st.session_state.expanded_tables[filename] = False

                is_expanded = st.session_state.expanded_tables[filename]

                with col2:
                    if len(df_mw) > 2:
                        btn_label = "一部隠す" if is_expanded else "すべて表示"
                        if st.button(btn_label, key=f"btn_{filename}"):
                            st.session_state.expanded_tables[filename] = not is_expanded
                            st.rerun()

                max_idx = None
                if "%" in df_mw.columns:
                    max_idx = df_mw['%'].idxmax()

                def styler_func(row):
                    styles = []
                    for col in row.index:
                        if row.name == max_idx and col in ['Mn', 'Ð']:
                            styles.append('background-color: yellow; color: black;')
                        else:
                            styles.append('color: black;')
                    return styles

                display_df = df_mw if is_expanded or len(df_mw) <= 2 else df_mw.head(2)
                st.dataframe(display_df.style.apply(styler_func, axis=1), hide_index=True, use_container_width=True)

        # --- Sidebar settings ---
        st.sidebar.header("🎨 グラフ設定")

        # 横軸表示範囲の手入力
        st.sidebar.subheader("横軸範囲の指定 (Retention time)")
        col_x_min, col_x_max = st.sidebar.columns(2)
        with col_x_min:
            x_range_min = st.number_input("最小値", value=15.0, step=1.0, format="%.1f", key="ui_x_min")
        with col_x_max:
            x_range_max = st.number_input("最大値", value=19.0, step=1.0, format="%.1f", key="ui_x_max")

        if x_range_min >= x_range_max:
            st.sidebar.error("最大値は最小値より大きい値を指定してください。")
            x_range_max = x_range_min + 1.0

        tick_interval = st.sidebar.number_input("横軸メモリの間隔", min_value=0.1, max_value=10.0, value=1.0, step=1.0, key="ui_tick_interval")

        st.sidebar.divider()
        show_yaxis = st.sidebar.checkbox("縦軸（強度数値）を表示", value=False, key="ui_show_yaxis")

        st.sidebar.subheader("凡例の設定")
        show_legend_ui = st.sidebar.checkbox("グラフ右上の凡例を表示", value=True, key="ui_show_legend")
        legend_x = st.sidebar.slider("凡例の横位置 (X座標)", min_value=0.0, max_value=1.1, value=st.session_state.get("legend_pos_x", 0.99), step=0.01, key="ui_legend_x")
        legend_y = st.sidebar.slider("凡例の縦位置 (Y座標)", min_value=0.0, max_value=1.1, value=st.session_state.get("legend_pos_y", 0.99), step=0.01, key="ui_legend_y")

        st.sidebar.divider()
        st.sidebar.header("📝 テキストメモ (アノテーション)")
        st.sidebar.markdown("グラフ上に任意のテキストを追加できます。")
        num_annotations = st.sidebar.number_input("追加するメモの数", min_value=0, max_value=10, value=0, step=1, key="ui_num_annotations")
        annotations = []
        for i in range(num_annotations):
            with st.sidebar.expander(f"メモ {i+1}"):
                show_ann = st.checkbox("このメモを表示する", value=True, key=f"ann_show_{i}")
                text = st.text_input(f"表示テキスト", value=f"Memo {i+1}", key=f"ann_text_{i}")
                font_size = st.number_input("文字サイズ", min_value=8, max_value=40, value=14, step=1, key=f"ann_s_{i}")
                color_ann = st.color_picker("文字色", value="#000000", key=f"ann_c_{i}")

                # 手動スライダーに変更
                ann_x = st.slider("X座標", min_value=-0.1, max_value=1.1, value=st.session_state.get(f"ann_pos_x_{i}", 0.5), step=0.01, key=f"ui_ann_x_{i}")
                ann_y = st.slider("Y座標", min_value=-0.1, max_value=1.1, value=st.session_state.get(f"ann_pos_y_{i}", 0.5), step=0.01, key=f"ui_ann_y_{i}")

                if show_ann:
                    # stateにも念のため同期させておく
                    st.session_state[f"ann_pos_x_{i}"] = ann_x
                    st.session_state[f"ann_pos_y_{i}"] = ann_y

                    annotations.append({
                        'idx': i,
                        'text': text,
                        'x': ann_x,
                        'y': ann_y,
                        'font_size': font_size,
                        'color': color_ann
                    })

        st.sidebar.divider()
        st.sidebar.header("💾 保存設定")
        img_format = st.sidebar.radio("カメラアイコンでの保存形式", ["svg", "png"], index=0, key="ui_img_format")

        st.sidebar.divider()
        st.sidebar.header("⚙ 各データの設定 (色・Y軸オフセット)")

        file_settings = {}

        for i, filename in enumerate(ordered_files):
            data = data_dict[filename]
            series_name = data['series_name']
            with st.sidebar.expander(f"設定: {series_name}"):
                custom_legend_name = st.text_input("凡例の表示名", value=st.session_state.get(f"legend_name_{filename}", series_name), key=f"legend_input_{filename}")
                st.session_state[f"legend_name_{filename}"] = custom_legend_name

                color = st.color_picker("線の色", value="#000000", key=f"color_{filename}")

                diff_y = 1.0 # Normalized range is 0 to 1, thus diff_y is approx 1
                max_y_offset = diff_y * 10.0 # 広めの操作可能範囲
                step_y = 0.1

                # 逆順をやめ、そのままのインデックスで負の方向にずらしていく（-1.5刻み）
                # 0番目のファイルが一番上(0.0)、1番目が下(-1.5)、2番目がさらに下(-3.0)となるようにする
                default_y = float(-i * 1.5)

                # スライダーのキーを `y_offset_pos_{i}` に固定することで、
                # 凡例の順番 1番目〜N番目 それぞれの高さ設定自体は維持され、
                # 中身のファイルだけが入れ替わる（期待通りのスワップ動作になる）
                slider_key = f"y_offset_pos_{i}"
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = default_y

                y_offset = st.slider("Y軸 オフセット (標準化後)", min_value=float(-max_y_offset), max_value=float(max_y_offset), value=float(st.session_state[slider_key]), step=step_y, key=slider_key)

                st.markdown("##### 📍 ピークトップ線設定")
                peak_enable = st.checkbox("ピークトップ（縦線）を表示", value=False, key=f"peak_enable_{filename}")
                peak_color = st.color_picker("線の色", value="#808080", key=f"peak_color_{filename}") # デフォルト灰色
                # dashの設定（実線がデフォルト）
                peak_dash = st.selectbox("線の種類", ["solid", "dash", "dot", "dashdot"], index=0, key=f"peak_dash_{filename}")

                file_settings[filename] = {
                    'color': color,
                    'y_offset': y_offset,
                    'peak_enable': peak_enable,
                    'peak_color': peak_color,
                    'peak_dash': peak_dash
                }

        # --- Plotting ---
        st.markdown("---")
        st.subheader("📈 クロマトグラム")
        fig = go.Figure()

        global_y_min = float('inf')
        global_y_max = float('-inf')

        traces_to_add = []
        shapes_to_add = []

        for filename in ordered_files:
            data = data_dict[filename]
            df = data['df_chrom'].copy()

            # 使用する凡例名（ユーザーが変更していればそれを適用）
            mapped_series_name = st.session_state.get(f"legend_name_{filename}", data['series_name'])

            settings = file_settings[filename]

            # 見えている範囲におけるグラフの最大値と最小値の差が等しくなるように標準化 (0 ~ 1)
            df = normalize_data(df, float(x_range_min), float(x_range_max))

            # SVGエクスポート等のベクター形式で、領域（枠線）の外まで線が長く飛び出すのを防ぐため
            # 表示範囲の「わずかに外側」でデータを物理的にカットする
            x_min_val = float(x_range_min)
            x_max_val = float(x_range_max)
            margin = (x_max_val - x_min_val) * 0.05

            plot_mask = (df["Retention time [min]"] >= x_min_val - margin) & (df["Retention time [min]"] <= x_max_val + margin)
            df_plot = df[plot_mask]

            x_data = df_plot["Retention time [min]"]
            y_data = df_plot["Intensity_Norm"] + settings['y_offset']

            c_min, c_max = y_data.min(), y_data.max()
            if c_min < global_y_min: global_y_min = c_min
            if c_max > global_y_max: global_y_max = c_max

            peak_x = None
            mask = (x_data >= float(x_range_min)) & (x_data <= float(x_range_max))
            if any(mask):
                max_y_idx = df_plot.loc[mask, "Intensity_Norm"].idxmax()
                peak_x = df_plot.loc[max_y_idx, "Retention time [min]"]
                peak_y = y_data.loc[max_y_idx]

            traces_to_add.append(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                line_shape='spline',
                name=mapped_series_name,
                line=dict(color=settings['color'], width=2.0),
                cliponaxis=True  # 領域外（SVGエクスポート時など）をクリッピング
            ))

            if settings['peak_enable'] and peak_x is not None:
                shapes_to_add.append({
                    'x': peak_x,
                    'y': peak_y,
                    'color': settings['peak_color'],
                    'dash': settings['peak_dash']
                })

        if global_y_min == float('inf'):
            global_y_min = 0.0
            global_y_max = 1.0

        for trace in traces_to_add:
            fig.add_trace(trace)

        for shape in shapes_to_add:
            # y0 を global_y_min に固定する。下のfig.update_yaxesでrange下端をglobal_y_minに合わせているため横軸とピッタリ合う
            fig.add_shape(
                type="line",
                x0=shape['x'], x1=shape['x'],
                y0=global_y_min, y1=shape['y'],
                yref="y", xref="x",
                line=dict(width=1.5, dash=shape['dash'], color=shape['color']),
                opacity=0.8
            )

        # 以前のUIから削除したグリッドと枠線の変数を固定値で定義
        show_grid = False
        show_border = False

        # 凡例の表示/非表示（ファイルの数が2つ以上かつ、UIのチェックがONのとき）
        show_legend_final = (len(data_dict) >= 2) and show_legend_ui

        # セッションに保存された凡例の位置（スライダーから）を適用
        st.session_state["legend_pos_x"] = legend_x
        st.session_state["legend_pos_y"] = legend_y

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=show_legend_final,
            legend=dict(
                yanchor="top", y=legend_y,
                xanchor="right", x=legend_x,
                font=dict(size=14, color="black", family="Arial, sans-serif"),
                bgcolor="rgba(255,255,255,0.7)" # 半透明の背景でグラフとの重なりを緩和
            )
        )

        # 凡例のテキスト自体を太文字にする場合は各traceのnameを太文字修飾する代わりに、
        # ここでtraceの名前を <b></b> で囲むことで対応（Plotly標準機能）
        for trace in fig.data:
            trace.name = f"<b>{trace.name}</b>"

        # アノテーション（テキストメモ）の描画
        for ann in annotations:
            fig.add_annotation(
                text=f"<b>{ann['text']}</b>", # アノテーションも太字
                xref="paper", yref="paper",   # グラフ領域全体を0~1と見なす相対座標系
                x=ann['x'], y=ann['y'],
                showarrow=False,
                font=dict(size=ann['font_size'], color=ann['color'])
            )

        # Plotlyで両端のメモリが切れないように描画範囲(range)を広げる
        range_padding = (x_range_max - x_range_min) * 0.01

        # 横軸のメモリ配列の計算
        x_ticks = np.arange(x_range_min, x_range_max + (tick_interval/1000), tick_interval)

        # numpy.arangeで生じた浮動小数点のゴミを丸めてフォーマット (%gは不要な0を省く)
        x_tick_texts = [f"<b>{val:g}</b>" for val in x_ticks]

        fig.update_xaxes(
            range=[x_range_min - range_padding, x_range_max + range_padding], # わずかに広げて両端メモリを保証
            tickmode='array',
            tickvals=x_ticks,
            ticktext=x_tick_texts,
            ticks='inside',
            tickcolor='black',
            tickwidth=2.5,
            ticklen=8,
            title_text="<b>Retention time [min]</b>",
            title_font=dict(size=15, color='black'),
            tickfont=dict(size=15, color='black'),
            showline=True,
            linecolor='black',
            linewidth=2.5,
            showgrid=show_grid,
            gridwidth=1,
            gridcolor='#e0e0e0',
            mirror=show_border,
            zeroline=False
        )

        # 下にはpaddingを持たせずぴったりに、上だけ余裕(0.05分)を持たせることで y=global_y_min が横軸ラインになる
        y_range_pad = (global_y_max - global_y_min) * 0.05
        if y_range_pad == 0: y_range_pad = 0.1

        fig.update_yaxes(
            range=[global_y_min, global_y_max + y_range_pad],
            showticklabels=show_yaxis,
            title_text="" if not show_yaxis else "<b>Normalized Intensity</b>",
            showline=show_border,
            linewidth=1,
            linecolor='black',
            showgrid=show_grid,
            gridwidth=1,
            gridcolor='#e0e0e0',
            mirror=show_border,
            zeroline=False
        )

        # Tracesを枠線にクリップ
        fig.update_traces(cliponaxis=True)

        # PNGとSVGの両方でダウンロードできるようにPlotlyのconfigを設定
        config = {
            'toImageButtonOptions': {
                'format': img_format,         # サイドバーで選択された形式
                'filename': 'gpc_plot',
                'height': 600,
                'width': 1000,
                'scale': 1
            },
            'edits': {
                'annotationPosition': True,
                'annotationText': False,
                'legendPosition': True       # 凡例のドラッグ移動を許可
            },
            'displaylogo': False,        # Plotlyロゴを非表示に
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'] # 任意に使えるよう少しおまけ
        }

        # Plotlyチャートを描画
        st.plotly_chart(fig, use_container_width=True, config=config)

# プロジェクト保存用のダウンロードボタンをサイドバー最下部に配置
if data_dict:
    st.sidebar.divider()
    st.sidebar.header("📦 プロジェクトの保存")

    def create_project_file():
        # Streamlitの内部情報やアップロード履歴などは除外する
        state_to_save = {k: v for k, v in st.session_state.items()
                         if not k.startswith("FormSubmitter")
                         and k != "last_loaded_project"
                         and not k.startswith("uploaded_")
                         and not k.startswith("up_")
                         and not k.startswith("down_")
                         and not k.startswith("btn_")}

        project_data = {
            'data_dict': data_dict,
            'session_state': state_to_save
        }
        return pickle.dumps(project_data)

    if len(data_dict) == 1:
        series_info = list(data_dict.values())[0]['series_name']
        dl_filename = f"{series_info}.gpc"
    else:
        # 複数ファイル時は最大3つまでの系列名繋ぎ合わせで作る
        series_names = [v['series_name'] for v in data_dict.values()]
        series_info = "_".join(series_names[:3])
        if len(series_names) > 3:
            series_info += "_etc"
        dl_filename = f"{series_info}.gpc"

    st.sidebar.download_button(
        label="📥 現在の状態を復元用ファイル(.gpc)として保存",
        data=create_project_file(),
        file_name=dl_filename,
        mime="application/octet-stream"
    )

