import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import t as t_dist
from scipy.optimize import curve_fit
import sqlite3
import hashlib
import time
from datetime import datetime
import matplotlib.pyplot as plt
import io
from contextlib import contextmanager

# ==================== 页面配置 ====================
st.set_page_config(page_title="小车实验分析助手 Pro", layout="wide")

# ==================== 数据库管理 ====================
DB_PATH = 'experiments.db'

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                drugs TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS data_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                recipe_id INTEGER,
                x REAL NOT NULL,
                y REAL NOT NULL,
                temperature REAL,
                record_time TEXT,
                upload_time TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(recipe_id) REFERENCES recipes(id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS fit_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                recipe_id INTEGER,
                model TEXT,
                equation TEXT,
                r2 REAL,
                x_points TEXT,
                y_points TEXT,
                fit_time TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(recipe_id) REFERENCES recipes(id)
            )
        ''')
        conn.commit()

init_db()

# ==================== 用户认证（简化版） ====================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = ""

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    with get_db() as conn:
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                         (username, hash_password(password)))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def login_user(username, password):
    with get_db() as conn:
        user = conn.execute('SELECT id, password FROM users WHERE username = ?',
                            (username,)).fetchone()
        if user and user['password'] == hash_password(password):
            return user['id']
        return None

# 侧边栏登录/注册
if not st.session_state.logged_in:
    st.sidebar.title("🔐 登录 / 注册")
    auth_mode = st.sidebar.radio("选择操作", ["登录", "注册"])
    username = st.sidebar.text_input("用户名")
    password = st.sidebar.text_input("密码", type="password")
    if auth_mode == "登录":
        if st.sidebar.button("登录"):
            user_id = login_user(username, password)
            if user_id:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.rerun()
            else:
                st.sidebar.error("用户名或密码错误")
    else:
        if st.sidebar.button("注册"):
            if register_user(username, password):
                st.sidebar.success("注册成功，请登录")
            else:
                st.sidebar.error("用户名已存在")
    st.stop()

# 已登录用户
st.sidebar.success(f"👤 当前用户：{st.session_state.username}")
if st.sidebar.button("退出登录"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = ""
    st.rerun()

# ==================== 多模型拟合器（原代码移植） ====================
class MultiModelFitter:
    MODEL_LABELS = {
        'linear': '线性 y=a·x+b',
        'inverse': '倒数线性 y=a/x+b',
        'power': '幂律 y=a·x^b (碘钟理论)',
        'quadratic': '二次多项式 y=a·x²+b·x+c',
        'log_model': '对数 y=a·ln(x)+b',
    }

    @classmethod
    def fit_all(cls, x_raw, y_raw):
        x = np.asarray(x_raw, float).ravel()
        y = np.asarray(y_raw, float).ravel()
        results = {}
        try:
            m = LinearRegression().fit(x.reshape(-1,1), y); y_p = m.predict(x.reshape(-1,1))
            k, b = float(m.coef_[0]), float(m.intercept_)
            results['linear'] = {'label':cls.MODEL_LABELS['linear'],'r2':r2_score(y,y_p),
                                 'eq':f'y = {k:.5f}·x + {b:.5f}','predict':lambda xv,k=k,b=b: k*np.asarray(xv,float)+b}
        except: pass
        try:
            mask = x!=0; xv,yv = x[mask],y[mask]
            if len(xv)>=2:
                x_inv = 1.0/xv; m = LinearRegression().fit(x_inv.reshape(-1,1), yv)
                y_p = m.predict(x_inv.reshape(-1,1)); k,b = float(m.coef_[0]),float(m.intercept_)
                results['inverse'] = {'label':cls.MODEL_LABELS['inverse'],'r2':r2_score(yv,y_p),
                                     'eq':f'y = {k:.5f}/x + {b:.5f}','predict':lambda xv,k=k,b=b: k/np.asarray(xv,float)+b}
        except: pass
        try:
            mask = (x>0)&(y>0); xv,yv = x[mask],y[mask]
            if len(xv)>=2:
                lx,ly = np.log(xv),np.log(yv); m = LinearRegression().fit(lx.reshape(-1,1), ly)
                b_exp = float(m.coef_[0]); a = float(np.exp(m.intercept_))
                y_p = a*xv**b_exp; r2 = r2_score(yv,y_p)
                results['power'] = {'label':cls.MODEL_LABELS['power'],'r2':r2,
                                    'eq':f'y = {a:.5f}·x^{b_exp:.5f}',
                                    'predict':lambda xv,a=a,b=b_exp: a*np.abs(np.asarray(xv,float))**b}
        except: pass
        try:
            if len(x)>=3:
                c = np.polyfit(x,y,2); y_p = np.polyval(c,x); a2,a1,a0 = c
                results['quadratic'] = {'label':cls.MODEL_LABELS['quadratic'],'r2':r2_score(y,y_p),
                                        'eq':f'y = {a2:.5f}x² + {a1:.5f}x + {a0:.5f}',
                                        'predict':lambda xv,c=c: np.polyval(c, np.asarray(xv,float))}
        except: pass
        try:
            mask = x>0; xv,yv = x[mask],y[mask]
            if len(xv)>=2:
                lx = np.log(xv); m = LinearRegression().fit(lx.reshape(-1,1), yv)
                y_p = m.predict(lx.reshape(-1,1)); k,b = float(m.coef_[0]),float(m.intercept_)
                results['log_model'] = {'label':cls.MODEL_LABELS['log_model'],'r2':r2_score(yv,y_p),
                                        'eq':f'y = {k:.5f}·ln(x) + {b:.5f}',
                                        'predict':lambda xv,k=k,b=b: k*np.log(np.abs(np.asarray(xv,float)))+b}
        except: pass
        return dict(sorted(results.items(), key=lambda kv: kv[1]['r2'], reverse=True))

# ==================== 主页面 ====================
st.title("🚗 小车实验分析助手 Pro")
st.markdown("碘钟反应 · 多模型拟合 · 在线协作")

# 初始化数据库
user_id = st.session_state.user_id

# 侧边栏：药品配方管理
with st.sidebar.expander("⚗️ 药品配方管理", expanded=False):
    # 获取用户的配方
    with get_db() as conn:
        recipes = conn.execute('SELECT id, name FROM recipes WHERE user_id = ?', (user_id,)).fetchall()
    recipe_names = [r['name'] for r in recipes] if recipes else []
    if recipe_names:
        selected_recipe = st.selectbox("选择配方", recipe_names, key="recipe_select")
        recipe_id = recipes[recipe_names.index(selected_recipe)]['id']
    else:
        selected_recipe = None
        recipe_id = None
        st.warning("暂无配方，请创建")
    new_recipe_name = st.text_input("新建配方名称")
    if st.button("创建配方") and new_recipe_name:
        with get_db() as conn:
            conn.execute('INSERT INTO recipes (user_id, name, drugs) VALUES (?, ?, ?)',
                         (user_id, new_recipe_name, "[]"))
            conn.commit()
        st.rerun()
    if recipe_id and st.button("删除选中配方"):
        with get_db() as conn:
            conn.execute('DELETE FROM recipes WHERE id = ? AND user_id = ?', (recipe_id, user_id))
            conn.commit()
        st.rerun()

# 主区域：数据点管理
if recipe_id:
    st.subheader(f"📊 实验数据点 - {selected_recipe}")
    # 加载该配方的数据点
    with get_db() as conn:
        points = conn.execute('SELECT x, y, temperature, record_time FROM data_points WHERE user_id = ? AND recipe_id = ?',
                              (user_id, recipe_id)).fetchall()
    df_points = pd.DataFrame(points, columns=['X', 'Y', '温度(℃)', '记录时间'])
    edited_df = st.data_editor(df_points, num_rows="dynamic", use_container_width=True,
                               column_config={
                                   "X": st.column_config.NumberColumn(required=True),
                                   "Y": st.column_config.NumberColumn(required=True),
                                   "温度(℃)": st.column_config.NumberColumn(format="%.1f"),
                                   "记录时间": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                               })
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存数据"):
            with get_db() as conn:
                conn.execute('DELETE FROM data_points WHERE user_id = ? AND recipe_id = ?', (user_id, recipe_id))
                for _, row in edited_df.iterrows():
                    if pd.notna(row['X']) and pd.notna(row['Y']):
                        conn.execute('INSERT INTO data_points (user_id, recipe_id, x, y, temperature, record_time) VALUES (?,?,?,?,?,?)',
                                     (user_id, recipe_id, row['X'], row['Y'], row['温度(℃)'] if pd.notna(row['温度(℃)']) else None,
                                      row['记录时间'].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row['记录时间']) else None))
                conn.commit()
            st.success("数据已保存")
    with col2:
        if st.button("📥 导入Excel"):
            uploaded = st.file_uploader("上传Excel/CSV", type=['xlsx','csv'])
            if uploaded:
                try:
                    if uploaded.name.endswith('.csv'):
                        imp_df = pd.read_csv(uploaded, header=None, names=['X','Y'])
                    else:
                        imp_df = pd.read_excel(uploaded, header=None, names=['X','Y'])
                    st.dataframe(imp_df)
                    if st.button("确认导入"):
                        with get_db() as conn:
                            for _, row in imp_df.iterrows():
                                conn.execute('INSERT INTO data_points (user_id, recipe_id, x, y) VALUES (?,?,?,?)',
                                             (user_id, recipe_id, row['X'], row['Y']))
                            conn.commit()
                        st.rerun()
                except Exception as e:
                    st.error(f"导入失败：{e}")

    # 拟合分析
    st.subheader("📈 拟合分析")
    if st.button("执行拟合 (多模型选择)"):
        x_data = df_points['X'].dropna().values
        y_data = df_points['Y'].dropna().values
        if len(x_data) < 2:
            st.warning("至少需要2个数据点")
        else:
            results = MultiModelFitter.fit_all(x_data, y_data)
            if not results:
                st.warning("没有可用的拟合结果")
            else:
                st.session_state.fit_results = results
                st.session_state.fit_x = x_data
                st.session_state.fit_y = y_data

    if 'fit_results' in st.session_state and st.session_state.fit_results:
        results = st.session_state.fit_results
        model_keys = list(results.keys())
        selected_model = st.selectbox("选择模型", model_keys, format_func=lambda k: results[k]['label'])
        res = results[selected_model]
        st.markdown(f"**方程**: {res['eq']}")
        st.markdown(f"**R²**: {res['r2']:.6f}")
        # 绘图
        fig, ax = plt.subplots()
        x_arr = np.asarray(st.session_state.fit_x, float)
        y_arr = np.asarray(st.session_state.fit_y, float)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_line = res['predict'](x_line)
        ax.scatter(x_arr, y_arr, color='#1e5abf', s=50, label='数据点')
        ax.plot(x_line, y_line, color='#00a3a3', lw=2, label=res['label'])
        ax.set_xlabel("X (浓度)"); ax.set_ylabel("Y (时间)")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 预测与反推
        st.subheader("🔮 预测与反推")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            pred_x = st.number_input("预测X值")
        with col_p2:
            if st.button("计算理论预测Y"):
                try:
                    y_pred = res['predict'](np.array([pred_x]))[0]
                    st.session_state.pred_y = y_pred
                    st.success(f"理论预测Y = {y_pred:.5f}")
                except Exception as e:
                    st.error(f"计算失败：{e}")
        with col_p3:
            distance = st.number_input("路程 (m)")
            speed = st.number_input("速度 (m/s)")
            if st.button("反推X") and speed > 0:
                target_time = distance / speed
                x_vals = np.linspace(0.001, 1000, 10000)
                y_vals = res['predict'](x_vals)
                idx = np.argmin(np.abs(y_vals - target_time))
                rev_x = x_vals[idx]
                st.info(f"所需时间={target_time:.2f}s，反推X≈{rev_x:.3f}")

        # 保存拟合记录
        if st.button("💾 保存当前拟合"):
            with get_db() as conn:
                conn.execute('INSERT INTO fit_history (user_id, recipe_id, model, equation, r2, x_points, y_points) VALUES (?,?,?,?,?,?,?)',
                             (user_id, recipe_id, selected_model, res['eq'], res['r2'],
                              str(x_arr.tolist()), str(y_arr.tolist())))
                conn.commit()
            st.success("拟合记录已保存")
else:
    st.info("👈 请先在侧边栏创建或选择一个配方")

# 侧边栏：计时器
with st.sidebar.expander("⏱ 实验计时器", expanded=False):
    if 'timer_start' not in st.session_state:
        st.session_state.timer_start = None
        st.session_state.timer_elapsed = 0.0
        st.session_state.laps = []
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("开始/停止"):
            if st.session_state.timer_start is None:
                st.session_state.timer_start = time.time()
            else:
                st.session_state.timer_elapsed += time.time() - st.session_state.timer_start
                st.session_state.timer_start = None
    with col_t2:
        if st.button("分段"):
            if st.session_state.timer_start is not None:
                lap = st.session_state.timer_elapsed + (time.time() - st.session_state.timer_start)
            else:
                lap = st.session_state.timer_elapsed
            st.session_state.laps.append(lap)
        if st.button("重置"):
            st.session_state.timer_start = None
            st.session_state.timer_elapsed = 0.0
            st.session_state.laps = []
    if st.session_state.timer_start is not None:
        current = st.session_state.timer_elapsed + (time.time() - st.session_state.timer_start)
    else:
        current = st.session_state.timer_elapsed
    s = int(current); ms = int((current - s)*10); m, s = divmod(s,60); h, m = divmod(m,60)
    st.markdown(f"### {h:02d}:{m:02d}:{s:02d}.{ms}")
    if st.session_state.laps:
        laps_text = ""
        prev = 0.0
        for i, lap in enumerate(st.session_state.laps, 1):
            diff = lap - prev
            s_lap = int(lap); ms_lap = int((lap - s_lap)*10); m_lap, s_lap = divmod(s_lap,60); h_lap, m_lap = divmod(m_lap,60)
            s_diff = int(diff); ms_diff = int((diff - s_diff)*10); m_diff, s_diff = divmod(s_diff,60); h_diff, m_diff = divmod(m_diff,60)
            laps_text += f"分段{i}: {h_lap:02d}:{m_lap:02d}:{s_lap:02d}.{ms_lap} (+{h_diff:02d}:{m_diff:02d}:{s_diff:02d}.{ms_diff})\n"
            prev = lap
        st.text(laps_text)

# 侧边栏：历史记录查看
with st.sidebar.expander("📜 拟合历史", expanded=False):
    if recipe_id:
        with get_db() as conn:
            history = conn.execute('SELECT model, equation, r2, fit_time FROM fit_history WHERE user_id = ? AND recipe_id = ? ORDER BY fit_time DESC',
                                   (user_id, recipe_id)).fetchall()
        if history:
            df_hist = pd.DataFrame(history, columns=['模型','方程','R²','保存时间'])
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.caption("暂无记录")

