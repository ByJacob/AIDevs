import glob
import json
import re

import pytesseract
import requests
from PIL import Image
from langfuse.decorators import observe
from neo4j import GraphDatabase

from models import *
from utils import find_flag, create_message, download_and_extract_zip, create_message_with_image
from tqdm import tqdm

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


def send_query(query):
    apidb = f"{domain}/apidb"
    body = {
        "task": "database",
        "apikey": mykey,
        "query": query
    }
    result = requests.post(apidb, json=body)
    return result.json()

class UserGraph:
    def __init__(self):
        uri = "neo4j://localhost"
        auth = ("neo4j", "neo4jneo4j")
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.clear_all()

    def close(self):
        self.driver.close()

    def clear_all(self):
        with self.driver.session() as session:
            session.write_transaction(self._delete_all_data)

    def add_user(self, username):
        with self.driver.session() as session:
            session.write_transaction(self._create_user_node, username)

    def add_connection(self, user1, user2):
        with self.driver.session() as session:
            session.write_transaction(self._create_connection, user1, user2)

    def find_shortest_path(self, user1, user2):
        with self.driver.session() as session:
            result = session.read_transaction(self._get_shortest_path, user1, user2)
            return result

    @staticmethod
    def _delete_all_data(tx):
        query = "MATCH (n) DETACH DELETE n"  # Deletes all nodes and relationships
        tx.run(query)
    @staticmethod
    def _create_user_node(tx, username):
        query = (
            "MERGE (u:User {name: $username}) "
            "RETURN u"
        )
        tx.run(query, username=username)

    @staticmethod
    def _create_connection(tx, user1, user2):
        query = (
            "MATCH (u1:User {name: $user1}), (u2:User {name: $user2}) "
            "MERGE (u1)-[:CONNECTIONS]->(u2)"
        )
        tx.run(query, user1=user1, user2=user2)

    @staticmethod
    def _get_shortest_path(tx, user1, user2):
        query = (
            "MATCH p = shortestPath((u1:User {name: $user1})-[:CONNECTIONS*]-(u2:User {name: $user2})) "
            "RETURN [node in nodes(p) | node.name] AS path"
        )
        result = tx.run(query, user1=user1, user2=user2)
        record = result.single()
        return record["path"] if record else None

def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )

    # URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"


    users = send_query("Select * from users")
    connections = send_query("Select * from connections")

    user_graph = UserGraph()
    user_ids = {}
    for user in users["reply"]:
        user_graph.add_user(user["username"])
        user_ids[user["id"]] = user["username"]
    for connection in connections["reply"]:
        user_graph.add_connection(user_ids[connection["user1_id"]], user_ids[connection["user2_id"]])
    result = user_graph.find_shortest_path("Rafał", "Barbara")
    pass
    answer = ", ".join(result)
    print(answer)
    data = dict(answer=answer, apikey=mykey, task="connections")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )


if __name__ == '__main__':
    main()
