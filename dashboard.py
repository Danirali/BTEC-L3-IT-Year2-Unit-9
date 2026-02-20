from flask import Flask

app = Flask(__name__)

@app.route('/')
def main_ui():
    return "<h1>Server Running...</h1>"

app.run('0.0.0.0',debug=True) 
