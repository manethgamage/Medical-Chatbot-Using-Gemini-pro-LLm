prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If the user greets greet back and if ask common question answer that.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else. just answer the question based on context
Helpful answer:
"""