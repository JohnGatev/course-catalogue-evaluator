import sqlite3
import json
import os
from datetime import datetime

DB_PATH = 'evaluations.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_name TEXT,
            timestamp TEXT,
            evaluation_summary TEXT,
            exact_quotes_json TEXT,
            email_english TEXT,
            email_dutch TEXT,
            original_content_type TEXT,
            original_content_path TEXT,
            instructor_email TEXT
        )
    ''')
    # Migrate: add columns for storing compliance table and original text
    for col in ('requirements_table_json TEXT', 'text_content TEXT'):
        try:
            c.execute(f'ALTER TABLE evaluations ADD COLUMN {col}')
        except Exception:
            pass
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_evaluation(course_name, evaluation_summary, exact_quotes, email_english, email_dutch,
                    content_type, content_path, instructor_email="",
                    requirements_table=None, text_content=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO evaluations (
            course_name, timestamp, evaluation_summary, exact_quotes_json,
            email_english, email_dutch, original_content_type, original_content_path,
            instructor_email, requirements_table_json, text_content
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        course_name,
        datetime.now().isoformat(),
        evaluation_summary,
        json.dumps(exact_quotes),
        email_english,
        email_dutch,
        content_type,
        content_path,
        instructor_email,
        json.dumps(requirements_table or []),
        text_content,
    ))
    eval_id = c.lastrowid
    conn.commit()
    conn.close()
    return eval_id

def get_all_evaluations():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM evaluations ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_evaluation(eval_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM evaluations WHERE id = ?', (eval_id,))
    conn.commit()
    conn.close()

def get_setting(key, default=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT value FROM settings WHERE key = ?', (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default

def save_setting(key, value):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, value))
    conn.commit()
    conn.close()

init_db()
