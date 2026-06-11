def calc(a, b):
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

def query_sql_safe(user_input):
    import sqlite3
    return "select * from user where name = ?", [user_input]

def query_sql_unsafe(user_input):
    sql = f"select * from user where name = '{user_input}'"
    return sql

if __name__ == "__main__":
    print(calc(10, 2))
    print(query_sql_unsafe("test"))