from flask import Flask, render_template, jsonify, request
import subprocess
import os
from scripts import option

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    try:
        # Path to your Python script
        script_path = 'scripts/option.py'
        data = request.get_json()  # Get JSON data from the request

        # Retrieve values from the request data
        input_text1 = data.get('inputText1')
        input_text2 = data.get('inputText2')
        
        # Call the function from external_script.py to update the variables
        option.calc(input_text1, input_text2)
    
        return jsonify({"status": "success", "message": "Script executed successfully"})
    
    except subprocess.CalledProcessError as e:
        # Handle script execution errors
        return jsonify({"status": "error", "error": str(e.stderr)}), 500
    
    except Exception as e:
        # Handle other errors
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/dateticker', methods=['POST'])
def dateticker():
    try:
        input_text1 = request.form.get('inputText1')
        input_text2 = request.form.get('inputText2')
        
        # Call the function from external_script.py to update the variables
        option.input(input_text1, input_text2)
        return jsonify({"status": "success", "message": "Updated successfully"})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)