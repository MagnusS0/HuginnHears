from langchain_core.runnables import RunnableLambda

def refine_summary(chain, refine_chain, split_docs):
    """
    Refines the summary of a document based on the split documents.

    Args:
        chain (object): The initial chain to be used if no existing answer is provided.
        refine_chain (object): The chain to be used if an existing answer is provided.
        split_docs (list): The list of documents to be processed.

    Returns:
        object: The final refined summary.

    """
    # Route the chains based on input
    def route(inputs):
        if "existing_answer" not in inputs or inputs["existing_answer"] is None:
            return chain
        return refine_chain

    final_chain = RunnableLambda(route)

    # Initialize the existing answer as None
    existing_answer = None

    # Loop through the chunks
    for doc in split_docs:
        input_data = {"text": doc, "existing_answer": existing_answer}
        result = final_chain.invoke(input_data)
        existing_answer = result

    return existing_answer