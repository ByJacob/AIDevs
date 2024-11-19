import glob
import re
import uuid

import pytesseract
import requests
from PIL import Image
from langfuse.decorators import observe
from qdrant_client.models import PointStruct

from embedding import *
from utils import *
from tqdm import tqdm

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    download_url = f"{domain}/dane/pliki_z_fabryki.zip"
    directory_path = os.path.join("tmp", "s03e02")
    download_and_extract_zip(directory_path, download_url)
    extract_zip(os.path.join(directory_path, "weapons_tests.zip "), os.path.join(directory_path, "weapons_tests"),
                pwd=1670)
    files_to_embedded = os.listdir(os.path.join(directory_path, "weapons_tests", "do-not-share"))
    model = OpenAiTextEmbedding3Large(debug=False)
    qdrant_collection_name = os.path.basename(__file__).split(".")[0]
    client_qdrant = init_qdrant(model, qdrant_collection_name)
    for file in tqdm(files_to_embedded, desc="embedding files"):
        with open(os.path.join(directory_path, "weapons_tests", "do-not-share", file), "r", encoding="utf-8") as f:
            content = f.read()
        content_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, content))
        try:
            client_qdrant.http.points_api.get_point(qdrant_collection_name, id=content_id)
        except Exception as e:
            embed = model.embedding(content)
            client_qdrant.upsert(
                collection_name=qdrant_collection_name,
                wait=True,
                points=[
                    PointStruct(id=content_id, vector=embed.embedding, payload={"date": file.split(".")[0]}),
                ]
            )
    embed_question = model.embedding("W raporcie, z którego dnia znajduje się wzmianka o kradzieży prototypu broni?")
    result = client_qdrant.search(
        qdrant_collection_name,
        query_vector=embed_question.embedding,
        limit=1
    )
    pass
    answer = result[0].payload["date"].replace("_", "-")
    print(answer)
    data = dict(answer=answer, apikey=mykey, task="wektory")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )


if __name__ == '__main__':
    main()
