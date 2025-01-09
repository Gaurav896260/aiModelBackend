from flask import Flask, request, jsonify
import os
import openai
from flask_cors import CORS
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from dotenv import load_dotenv

app = Flask(__name__)

# Allow CORS for specified origins
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

load_dotenv()

# Set up OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

NOTEBOOK_PATH = r'D:\G1_UserInterface\KEMP_Model\CRS_KG\KG.ipynb'

class NotebookExecutor:
    def __init__(self, notebook_path):
        self.notebook_path = notebook_path
        with open(notebook_path, 'r', encoding='utf-8') as f:
            self.notebook = nbformat.read(f, as_version=4)

    def modify_input_cell(self, cell, user_input):
        """Replace input() calls with the user's input value."""
        if 'input(' in cell.source:
            # Replace the input function call with a direct string assignment
            cell.source = cell.source.replace('input("chat: ")', f'"{user_input}"')
        return cell

    def execute_tagged_cells(self, start_tag, end_tag, user_input):
        """Execute cells between start_tag and end_tag, replacing input() with user_input."""
        # Find the tagged cells
        start_index = end_index = None
        for i, cell in enumerate(self.notebook.cells):
            if 'tags' in cell.metadata:
                if start_tag in cell.metadata.tags and start_index is None:
                    start_index = i
                if end_tag in cell.metadata.tags:
                    end_index = i + 1  # Include the end tag cell

        if start_index is None or end_index is None:
            raise ValueError(f"Cells with tags '{start_tag}' or '{end_tag}' not found")

        # Create a new notebook with the selected cells
        execution_notebook = nbformat.v4.new_notebook()
        execution_notebook.cells = self.notebook.cells[start_index:end_index]

        # Modify the input cell
        for cell in execution_notebook.cells:
            if cell.cell_type == 'code':
                self.modify_input_cell(cell, user_input)

        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(execution_notebook, {'metadata': {'path': os.path.dirname(self.notebook_path)}})
            return execution_notebook
        except Exception as e:
            raise RuntimeError(f"Error executing notebook: {str(e)}")

    def get_output(self, executed_notebook):
        """Extract output from the last cell with output."""
        output_content = []
        for cell in reversed(executed_notebook.cells):
            if cell.outputs:
                for output in cell.outputs:
                    if 'text' in output:
                        output_content.append(output['text'])
                    elif 'data' in output and 'text/plain' in output['data']:
                        output_content.append(output['data']['text/plain'])
                if output_content:
                    break
        return '\n'.join(output_content)

@app.route('/api/suggestions', methods=['POST'])
def get_movie_suggestions():
    """Route for generating movie suggestions by executing the Jupyter notebook."""
    data = request.json
    user_input = data.get('query')
    
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    try:
        executor = NotebookExecutor(NOTEBOOK_PATH)
        executed_notebook = executor.execute_tagged_cells('first_cell', 'last_cell', user_input)
        output = executor.get_output(executed_notebook)
        
        if not output:
            return jsonify({"error": "No output generated"}), 500

        return jsonify({'output': output.strip()})

    except Exception as e:
        print(f"Error executing notebook: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation', methods=['POST'])
def conversation():
    """Route for simple conversation responses."""
    data = request.get_json()
    user_query = data.get('query', '').lower()

    responses = {
        'hello': "Hello! How can I help you today?",
        'hey': "Hello! How can I help you today?",
        'hi': "Hello! How can I help you today?",
        'how are you': "I'm doing well, thank you for asking! How about you?",
        'bye': "Goodbye! Have a great day!",
        'what is your name': "I'm movie recommender chatbot! I'm here to help you with anything you need. How can I assist you?"
    }

    return jsonify(response=responses.get(user_query, 
        "I'm not sure how to respond to that. Can you please rephrase or ask something else?"))

if __name__ == '__main__':
    app.run(debug=True, port=5001)