from flask import Flask, render_template

# Create Flask app
app = Flask(__name__)

# Define route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a simple API route
@app.route('/api/hello')
def api_hello():
    return {"message": "Hello, World!"}

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
