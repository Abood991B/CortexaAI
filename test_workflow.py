from src.workflow import process_prompt_with_langgraph

# Test the workflow with a simple prompt
test_prompt = "Write a function to sort a list of numbers"

try:
    result = process_prompt_with_langgraph(test_prompt, "auto")
    print("Workflow successful:")
    print(f"Status: {result.get('status')}")
    print(f"Domain: {result.get('output', {}).get('domain')}")
    print(f"Optimized Prompt: {result.get('output', {}).get('optimized_prompt')}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
