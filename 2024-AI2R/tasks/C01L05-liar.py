from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from base import BaseTasks


class Task(BaseTasks):
    def __init__(self):
        super().__init__("liar")

    def resolve(self, task):
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        extra_task = self.task_post(dict(question="what color is the sun?"))
        print("Extra: ", extra_task)
        system_template = """
            You are the verifier. You verify that the given sentence is true. You answer only YES or NO
        """
        human_template = "{text}"
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        prompt = chat_prompt.format_messages(text=extra_task)
        output = llm.invoke(prompt)
        return output.content


if __name__ == "__main__":
    t = Task()
    t.process()
