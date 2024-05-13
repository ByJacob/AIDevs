from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from base import BaseTasks


class Task(BaseTasks):
    def __init__(self):
        super().__init__("blogger")

    def resolve(self, task):
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        system_template = """
                You are blogger. Write some article about given topic. Article write in Polish language.
                """
        human_template = "{topic}"
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        results = []
        for topic in task["blog"]:
            prompt = chat_prompt.format_messages(topic=topic)
            output = llm.invoke(prompt)
            results.append(output.content)
        return results


if __name__ == "__main__":
    t = Task()
    t.process()
