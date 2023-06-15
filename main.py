from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field, conlist

load_dotenv()

model_name = 'text-davinci-003'
llm = OpenAI(model_name=model_name, temperature=0)


class MusicTasteDescriptionResult(BaseModel):
    genres: Optional[conlist(str, min_items=1, max_items=5)] =  Field(description="Music genres liked by the user. Must contain between 1 and 5 genres")
    bands: Optional[conlist(str, min_items=1, max_items=5)] = Field(description="Specific bands or artists liked by the user. If provided, must contain between 1 and 5 bands or artists")
    albums: Optional[conlist(str, min_items=1, max_items=5)] = Field(description="Specific albums liked by the user. If provided, must contain between 1 and 5 albums")
    year_range: Optional[conlist(int, min_items=2, max_items=2)] = Field(description="Year range of music liked by the user. If provided, must contain exactly 2 years indicating the start and end of the range")

parser = PydanticOutputParser(pydantic_object=MusicTasteDescriptionResult)

examples = [
    {
        "music taste description": "I like rock such as Rolling Stones or The Ramones, or the album London Calling from the clash",
        "result": MusicTasteDescriptionResult.parse_obj({
            "genres": ["rock"],
            "bands": ["Rolling Stones", "The Ramones", "The Clash"],
            "albums": ["London Calling"]
        }).json().replace("{", "{{").replace("}", "}}"),
    },
    {
        "music taste description": "I enjoy rock music from the 70s like Led Zeppelin",
        "result": MusicTasteDescriptionResult.parse_obj({
            "genres": ["rock"],
            "bands": ["Led Zeppelin"],
            "year_range": [1970, 1979]
        }).json().replace("{", "{{").replace("}", "}}"),
    },
]

example_prompt = PromptTemplate(input_variables=["music taste description", "result"], template="Query: {music taste description}\nResult:\n{result}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt, 
    prefix="""Given a query describing a user's music taste, transform it into a structured object.
    {format_instructions}
    """,
    suffix="Query: {input}\nResult:\n", 
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

print(example_prompt.format(**examples[0])) 

print("\n#######\n")

print(prompt.format(input="My favorite band is The Beatles"))

chain = LLMChain(llm=llm, prompt=prompt)

output = chain.run('I love pop music from the 80s, especially Madonna')
print(output)

try:
    parsed_taste = parser.parse(output)
    print(f"""
        Genres: {", ".join(parsed_taste.genres) if parsed_taste.genres else 'Not specified'}
        Bands: {", ".join(parsed_taste.bands) if parsed_taste.bands else 'Not specified'}
        Albums: {", ".join(parsed_taste.albums) if parsed_taste.albums else 'Not specified'}
        Year Range: {f"{parsed_taste.year_range[0]} - {parsed_taste.year_range[1]}" if parsed_taste.year_range else 'Not specified'}
    """)
except Exception as e:
    print(e)