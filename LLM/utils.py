import boto3
import json
from litellm import completion
from IPython.display import Markdown
import concurrent.futures
import json
from IPython.display import display, HTML
import ipywidgets as widgets

session = boto3.Session()
bedrock = session.client('bedrock-runtime', region_name='us-east-1')


def call_bedrock_claude(question , bedrock = bedrock):
            
    response = completion(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{ "content": question,"role": "user"}],
      
            aws_bedrock_client= bedrock,
            max_tokens = 200000
    )
    return response['choices'][0]['message']['content']



def run_parallel_calls(inputs):

    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = list(executor.map(call_bedrock_claude, inputs))

    return results


def datasets(path = './data_examples'):
    
    train_indices = [0, 1, 2, 9, 3, 6]
    train_dicts = []
    test_dicts = []
    for i in range(11):
        with open(f"{path}/data_{i}.json", "r") as file:
            data = json.load(file)
        if i in train_indices:
            train_dicts.append(data)
        else:
            test_dicts.append(data)
            
    return train_dicts, test_dicts

        

        
def generate_config(description, train_dicts, test_dicts ):
        
    test_idx = 0
    question = f"""
    Using the following examples, return only the task's output in JSON format:

    Example 1:
    Input:
    {train_dicts[0]['Input']}
    Output:
    {json.dumps(train_dicts[0]['Output'])}
    Example 2:
    Input:
    { train_dicts[1]['Input']}
    Output:
    {json.dumps(train_dicts[1]['Output'])}
    Example 3:
    Input:
    {train_dicts[2]['Input']}
    Output:
    {json.dumps(train_dicts[2]['Output'])}
    Example 4:
    Input:
    {train_dicts[3]['Input']}
    Output:
    {json.dumps(train_dicts[3]['Output'])}
    Example 5:
    Input:
    {train_dicts[4]['Input']}
    Output:
    {json.dumps(train_dicts[4]['Output'])}
    Task:
    Input:
    {test_dicts[test_idx]['Input']}
    Output:

    """
    question_list = [question] * 3
    results = run_parallel_calls(question_list)
    
    question2 = f"""
    Given the following list of generated JSON objects, return the JSON object that has the highest agreement or consensus among the responses. 
    Return only the JSON object without any additional text:

    {results}
    """
    
    question_list2 = [question2] * 3
    final_outputs = run_parallel_calls(question_list2 )
    for output in final_outputs:
        try:
            config_json = json.loads(output)
            return config_json
        except:
            pass
    
    output_message = f"""The following config generated but JSON parsing failed and interactive mode will not work:
    {output}
    """
    return output_message
    

def display_config(data, file_path):
 
    # Create a textarea widget to display the JSON
    json_text = widgets.Textarea(
        value=json.dumps(data, indent=4),
        layout={'height': '500px', 'width': '100%'}
    )
    
    # Create a button widget to save JSON
    save_button = widgets.Button(description='Save Config')
    
    def save_json(change):
        try:
            updated_json = json.loads(json_text.value)
        except json.JSONDecodeError as e:
            json_text.value = str(e)
        else:
            with open(file_path, 'w') as json_file:
                json.dump(updated_json, json_file, indent=4)
            json_text.value = "Config saved successfully!"
    
    save_button.on_click(save_json)
    
    # Display the widgets
    display(json_text)
    display(save_button)