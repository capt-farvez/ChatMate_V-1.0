from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

model_path = "C:\\Users\\ANWAR HUSSAIN PARVAI\\Documents\\Orca_Mini_Model_3B\\orca-mini-3b.ggmlv3.q4_0.bin"

def generate_text(question):
    template="""
	Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request in 50 words.

	### Instruction:
	Instruction

	### Input:
	{question}

	### Response:

    """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    callbacks = [StreamingStdOutCallbackHandler()]

    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=model_path, callbacks=callbacks, verbose=True)

    # If you want to use a custom model add the backend parameter
    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
    llm = GPT4All(model=model_path, backend="gptj", callbacks=callbacks, verbose=True)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer=llm_chain.run(question)
    return answer

if __name__ == "__main__":
    print("Chatbot: Hello! Type 'exit' to end the conversation.")
    while True:
        user_input = input("User    : ")
        if user_input.lower() == "exit":
            print("Chatbot : Goodbye! Have a nice Day.")
            break
        response = generate_text(user_input)
        print("Chatbot:", response)
        print()