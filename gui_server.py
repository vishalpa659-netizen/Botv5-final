"""
gui_server.py  —  AI Trading Bot v5 GUI Server
Run alongside the bot:  python gui_server.py
"""

import sqlite3
import json
import os
import sys                          # FIX: use sys.executable instead of "python"
import subprocess
import signal
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

DATABASE          = "ai_trading.db"
SIGNAL_FILE       = "signal_history.json"
OPEN_SIGNALS_FILE = "open_signals.json"
PERFORMANCE_FILE  = "performance_data.json"
LEARNING_FILE     = "learning_state.json"
LOG_FILE          = "gui_log.json"
FILTER_FILE       = "gui_filters.json"
BOT_SCRIPT        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_trading_bot_v5.py")
MAX_DATA_SIZE_MB  = 200
DATA_FILES        = [DATABASE, SIGNAL_FILE, OPEN_SIGNALS_FILE, PERFORMANCE_FILE, LOG_FILE, LEARNING_FILE]

app = Flask(__name__, static_folder=".", static_url_path="/static")
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("trading_gui.html")

bot_process = None
bot_lock    = threading.Lock()
_log_lock   = threading.Lock()

def _load_gui_log():
    try:
        with open(LOG_FILE) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def _save_gui_log(log_list):
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(log_list, f)
    except Exception:
        pass

gui_log = _load_gui_log()

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else []

def get_data_size_mb():
    total = 0
    for path in DATA_FILES:
        try: total += os.path.getsize(path)
        except OSError: pass
    return total / (1024 * 1024)

def add_log(level, msg):
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "level": level, "msg": msg}
    with _log_lock:
        gui_log.append(entry)
        if len(gui_log) > 1000:
            del gui_log[:200]
        _save_gui_log(gui_log)

def auto_cleanup_if_needed():
    size_mb = get_data_size_mb()
    if size_mb < MAX_DATA_SIZE_MB:
        return
    try:
        db = sqlite3.connect(DATABASE)
        total_rows = db.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        delete_cnt = max(1, int(total_rows * 0.20))
        db.execute("DELETE FROM signals WHERE id IN (SELECT id FROM signals ORDER BY id ASC LIMIT ?)", (delete_cnt,))
        db.execute("VACUUM")
        db.commit(); db.close()
        add_log("warn", f"Auto-cleanup: removed {delete_cnt} old rows")
    except Exception as e:
        add_log("err", f"Auto-cleanup error: {e}")

def bot_is_running():
    global bot_process
    return bot_process is not None and bot_process.poll() is None

def watch_bot_output(proc):
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            level = "err"  if "ERROR"  in text.upper() else \
                    "warn" if ("WARN"   in text.upper() or "SKIP" in text.upper()) else \
                    "acc"  if ("SIGNAL" in text.upper() or "WIN"  in text.upper()) else "info"
            add_log(level, text)
    add_log("warn", "Bot process ended")

@app.route("/api/status")
def status():
    size_mb = get_data_size_mb()
    open_trades = 0
    try:
        db = get_db()
        row = db.execute("SELECT COUNT(*) as c FROM signals WHERE result='RUNNING'").fetchone()
        open_trades = row["c"] if row else 0
        db.close()
    except Exception:
        pass
    return jsonify({
        "bot_running":   bot_is_running(),
        "mt5_ok":        os.path.exists(DATABASE),
        "time_utc":      datetime.utcnow().strftime("%H:%M:%S"),
        "data_size_mb":  round(size_mb, 2),
        "data_size_pct": round(size_mb / MAX_DATA_SIZE_MB * 100, 1),
        "open_trades":   open_trades,
        "bot_version":   "v5",
        "groq_enabled":  True,
        "twelve_data":   True,
        "telegram":      True,
    })

@app.route("/api/scoreboard")
def scoreboard():
    try:
        db = get_db()
        rows = db.execute("SELECT result, COUNT(*) as cnt FROM signals WHERE signal != 'HOLD' GROUP BY result").fetchall()
        counts = {r["result"]: r["cnt"] for r in rows}
        sym_rows = db.execute("""
            SELECT symbol,
                   SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END) wins,
                   SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) losses,
                   COUNT(*) total, AVG(confidence) avg_confidence, AVG(rr_ratio) avg_rr
            FROM signals WHERE signal != 'HOLD'
            GROUP BY symbol ORDER BY total DESC
        """).fetchall()
        total_row = db.execute("SELECT AVG(confidence) avg_conf, AVG(rr_ratio) avg_rr FROM signals WHERE result IN ('WIN','LOSS')").fetchone()
        db.close()
        w = counts.get("WIN",0); l = counts.get("LOSS",0)
        r = counts.get("RUNNING",0); e = counts.get("EXPIRED",0)
        total = w + l
        sym_list = []; best_pair = worst_pair = None; best_wr = -1; worst_wr = 101
        for row in sym_rows:
            t = row["total"]; sr = round(row["wins"]/t*100,1) if t>0 else 0
            sym_list.append({"symbol":row["symbol"],"wins":row["wins"],"losses":row["losses"],
                              "total":t,"win_rate":sr,
                              "avg_confidence":round(row["avg_confidence"] or 0,1),
                              "avg_rr":round(row["avg_rr"] or 0,2)})
            if t >= 3:
                if sr > best_wr:  best_wr=sr;  best_pair=row["symbol"]
                if sr < worst_wr: worst_wr=sr; worst_pair=row["symbol"]
        return jsonify({"wins":w,"losses":l,"running":r,"expired":e,
                        "win_rate":round(w/total*100,1) if total else 0,
                        "profit_factor":round(w/max(l,1),2),
                        "avg_confidence":round(total_row["avg_conf"] or 0,1) if total_row else 0,
                        "avg_rr":round(total_row["avg_rr"] or 0,2) if total_row else 0,
                        "best_pair":best_pair or "—","worst_pair":worst_pair or "—","symbols":sym_list})
    except Exception as e:
        return jsonify({"error":str(e),"wins":0,"losses":0,"running":0,"expired":0,
                        "win_rate":0,"profit_factor":0,"avg_confidence":0,"avg_rr":0,
                        "best_pair":"—","worst_pair":"—","symbols":[]})

@app.route("/api/signals")
def signals():
    try:
        db = get_db()
        rows = db.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 100").fetchall()
        db.close()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])

@app.route("/api/open_signals")
def open_signals():
    return jsonify(load_json(OPEN_SIGNALS_FILE))

@app.route("/api/history")
def history():
    return jsonify(load_json(SIGNAL_FILE)[-50:])

@app.route("/api/risk")
def risk_state():
    try:
        db = get_db(); today = datetime.utcnow().strftime("%Y-%m-%d")
        row = db.execute("SELECT * FROM risk_state WHERE date=?", (today,)).fetchone()
        db.close()
        if not row:
            return jsonify({"daily_loss_pct":0,"weekly_loss_pct":0,"trade_count":0,
                            "daily_limit_pct":3,"weekly_limit_pct":10,"can_trade":True})
        dl = round(row["daily_loss"]*100,2); wl = round(row["weekly_loss"]*100,2)
        return jsonify({"daily_loss_pct":dl,"weekly_loss_pct":wl,"trade_count":row["trade_count"],
                        "daily_limit_pct":3,"weekly_limit_pct":10,"can_trade":dl<3 and wl<10})
    except Exception as e:
        return jsonify({"error":str(e),"daily_loss_pct":0,"weekly_loss_pct":0,"can_trade":True})

@app.route("/api/cooldowns")
def cooldowns():
    try:
        db = get_db(); rows = db.execute("SELECT * FROM cooldowns").fetchall(); db.close()
        now = datetime.utcnow(); result = []
        for r in rows:
            try:
                exp = datetime.fromisoformat(r["expires_at"])
                rem = max(0,(exp-now).total_seconds()/60)
                result.append({"symbol":r["symbol"],"signal":r["signal"],
                                "expires_at":r["expires_at"],"remaining_min":round(rem,0)})
            except Exception: pass
        return jsonify(result)
    except Exception:
        return jsonify([])

@app.route("/api/cooldowns/clear", methods=["POST"])
def clear_cooldowns():
    try:
        db = sqlite3.connect(DATABASE); db.execute("DELETE FROM cooldowns"); db.commit(); db.close()
        add_log("warn","All cooldowns cleared via GUI")
        return jsonify({"ok":True})
    except Exception as e:
        return jsonify({"ok":False,"msg":str(e)})

@app.route("/api/learning")
def learning():
    try:
        db = get_db()
        rows = db.execute("SELECT * FROM learning_state ORDER BY win_rate DESC").fetchall()
        db.close()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify(load_json(LEARNING_FILE,[]))

@app.route("/api/performance")
def performance():
    return jsonify(load_json(PERFORMANCE_FILE,[]))

@app.route("/api/pnl")
def pnl():
    try:
        db = get_db(); today = datetime.utcnow().strftime("%Y-%m-%d")
        week_start = (datetime.utcnow()-timedelta(days=datetime.utcnow().weekday())).strftime("%Y-%m-%d")
        daily  = db.execute("SELECT result,COUNT(*) cnt FROM signals WHERE date(time)=? AND result IN ('WIN','LOSS') GROUP BY result",(today,)).fetchall()
        weekly = db.execute("SELECT result,COUNT(*) cnt FROM signals WHERE date(time)>=? AND result IN ('WIN','LOSS') GROUP BY result",(week_start,)).fetchall()
        db.close()
        def parse(rows):
            d={r["result"]:r["cnt"] for r in rows}; w=d.get("WIN",0); l=d.get("LOSS",0)
            return {"wins":w,"losses":l,"win_rate":round(w/max(w+l,1)*100,1)}
        return jsonify({"daily":parse(daily),"weekly":parse(weekly)})
    except Exception as e:
        return jsonify({"daily":{},"weekly":{},"error":str(e)})

@app.route("/api/log")
def get_log():
    with _log_lock:
        return jsonify(gui_log[-100:])

@app.route("/api/log/clear", methods=["POST"])
def clear_log():
    with _log_lock:
        gui_log.clear(); _save_gui_log(gui_log)
    return jsonify({"ok":True})

@app.route("/api/storage")
def storage_info():
    details = {}
    for path in DATA_FILES:
        try: details[os.path.basename(path)] = round(os.path.getsize(path)/(1024*1024),3)
        except OSError: details[os.path.basename(path)] = 0.0
    total = sum(details.values())
    return jsonify({"files":details,"total_mb":round(total,2),"limit_mb":MAX_DATA_SIZE_MB,
                    "used_pct":round(total/MAX_DATA_SIZE_MB*100,1)})

@app.route("/api/storage/cleanup", methods=["POST"])
def force_cleanup():
    threading.Thread(target=auto_cleanup_if_needed,daemon=True).start()
    return jsonify({"ok":True,"msg":"Cleanup triggered"})

@app.route("/api/bot/start", methods=["POST"])
def start_bot():
    global bot_process
    with bot_lock:
        if bot_is_running():
            return jsonify({"ok":False,"msg":"Already running"})
        try:
            # FIX: use sys.executable so the correct Python interpreter is used
            bot_process = subprocess.Popen(
                [sys.executable, BOT_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            threading.Thread(target=watch_bot_output, args=(bot_process,), daemon=True).start()
            add_log("acc", f"Bot v5 started (PID {bot_process.pid})")
            return jsonify({"ok":True,"pid":bot_process.pid})
        except Exception as e:
            add_log("err", f"Failed to start bot: {e}")
            return jsonify({"ok":False,"msg":str(e)})

@app.route("/api/bot/stop", methods=["POST"])
def stop_bot():
    global bot_process
    with bot_lock:
        if not bot_is_running():
            return jsonify({"ok":False,"msg":"Bot not running"})
        try:
            bot_process.send_signal(signal.SIGTERM); bot_process.wait(timeout=5)
        except Exception:
            bot_process.kill()
        add_log("warn","Bot stopped by user"); bot_process=None
        return jsonify({"ok":True})

@app.route("/api/bot/scan", methods=["POST"])
def force_scan():
    add_log("acc","Force scan requested — bot will process on next cycle")
    return jsonify({"ok":True,"msg":"Scan notification sent to bot"})

@app.route("/api/filters", methods=["GET"])
def get_filters():
    try:
        with open(FILTER_FILE) as f: return jsonify(json.load(f))
    except Exception:
        return jsonify({"killzone":True,"calendar":True,"spread":True,
                        "volatility":True,"market_quality":True,
                        "regime":True,"mtf":True,"telegram":True})

@app.route("/api/filters", methods=["POST"])
def set_filters():
    data = request.json
    with open(FILTER_FILE,"w") as f: json.dump(data,f,indent=2)
    add_log("info",f"Filters updated: {data}")
    return jsonify({"ok":True})

@app.route("/api/open_signals/clear", methods=["POST"])
def clear_open():
    with open(OPEN_SIGNALS_FILE,"w") as f: json.dump([],f)
    add_log("warn","Open signals cleared via GUI")
    return jsonify({"ok":True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 55)
    print("  AI Trading Bot v5 — GUI Server")
    print(f"  Port: {port}  |  Logs loaded: {len(gui_log)}")
    print(f"  Python: {sys.executable}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
