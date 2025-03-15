from flask import Flask, render_template, jsonify, request
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    try:
        # Path to your Python script
        script_path = 'scripts/option.py'
        
        # Run the Python script
        result = subprocess.run(['python', script_path], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        
        # Return success response
        return jsonify({"status": "success", "message": "Script executed successfully"})
    
    except subprocess.CalledProcessError as e:
        # Handle script execution errors
        return jsonify({"status": "error", "error": str(e.stderr)}), 500
    
    except Exception as e:
        # Handle other errors
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)