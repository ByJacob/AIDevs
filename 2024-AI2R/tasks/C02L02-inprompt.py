from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import difflib

from base import BaseTasks


class Task(BaseTasks):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        super().__init__("inprompt")

    def get_name(self, text):
        system_template = "Return name from text. Don't return anything others."
        human_template = "{text}"
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        prompt = chat_prompt.format_messages(text=text)
        output = self.llm.invoke(prompt)
        return output.content

    def filter_intput(self, input, name):
        result = []
        cutoff = 0.9
        while len(result) <= 0:
            for i in input:
                sim = difflib.get_close_matches(name, i.split(" "), cutoff=cutoff)
                if len(sim) > 0:
                    result.append(i)
            cutoff -= 0.2
        return result

    def answer_question(self, question, content):
        system_template = "Answer to question using givent content\n ```content\n{content}```"
        human_template = "{question}"
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        prompt = chat_prompt.format_messages(question=question, content=content)
        output = self.llm.invoke(prompt)
        return output.content

    def resolve(self, task):
        input = task["input"]
        question = task["question"]
        name = self.get_name(question)
        filtered = self.filter_intput(input, name)
        answer = self.answer_question(question, ". ".join(filtered))
        return answer


if __name__ == "__main__":
    t = Task()
    t.process()
