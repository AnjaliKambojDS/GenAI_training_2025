import mlflow
from langchain.llms import VertexAI
# from langchain.tools import YahooFinanceSymbolSuggestTool, BraveSearchTool
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain.chains import SequentialChain, SimpleSequentialChain
from langchain import LLMChain, PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from google import genai
from google.genai.types import HttpOptions

# def get_stock_code(company_name: str) -> str:
    
#     tool = YahooFinanceSymbolSuggestTool()
#     result = tool.run(company_name)
#     # Extract and return stock ticker from result
#     return result['symbol']
def get_stock_code(company_name: str) -> str:
    """ Lookup stock code for a given company name using a free API.
    Returns stock code as string; returns "UNKNOWN" if not found.
    """
    try:
        url = f"https://stock-symbol-lookup-api.onrender.com/{company_name}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("stock_symbol", "UNKNOWN")
        else:
            print(f"Warning: Failed to fetch stock symbol for {company_name}: HTTP {response.status_code}")
            return "UNKNOWN"
    except Exception as e:
        print(f"Error during stock symbol lookup: {e}")
        return "UNKNOWN"
        
# def fetch_news(stock_code: str) -> list[str]:
    
#     tool = BraveSearchTool(max_results=5)
#     news_results = tool.run(f"{stock_code} stock latest news")
#     # Extract headlines or summaries
#     return [item['headline'] for item in news_results]

def fetch_news(stock_code: str) -> list:
    """
    Use LangChain DuckDuckGoSearchRun tool to fetch latest news headlines for a stock symbol.
    Returns a list of concise news headlines or summaries.
    """
    search = DuckDuckGoSearchRun()
    query = f"{stock_code} stock latest news"
    search_results = search.invoke(query)  # Runs the DuckDuckGo search
    
    # Split results into lines and extract headlines (for simplicity)
    results_lines = search_results.split('\n')
    # Filter out empty lines or irrelevant lines, limit to top 5
    headlines = [line.strip() for line in results_lines if line.strip()][:5]
    
    return headlines

# Example usage:
if __name__ == "__main__":
    stock_symbol = "AAPL"
    headlines = fetch_news(stock_symbol)
    print(f"Latest news headlines for {stock_symbol}:")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")


def analyze_sentiment(news_summaries: list[str], company_name: str, stock_code: str):
    # Define expected output schema
    response_schemas = [
        ResponseSchema(name="company_name", description="The company name"),
        ResponseSchema(name="stock_code", description="The stock ticker"),
        ResponseSchema(name="newsdesc", description="Summary of news"),
        ResponseSchema(name="sentiment", description="Sentiment: Positive/Negative/Neutral"),
        ResponseSchema(name="people_names", description="People mentioned"),
        ResponseSchema(name="places_names", description="Places mentioned"),
        ResponseSchema(name="other_companies_referred", description="Other companies mentioned"),
        ResponseSchema(name="related_industries", description="Industries related to news"),
        ResponseSchema(name="market_implications", description="Market impact summary"),
        ResponseSchema(name="confidence_score", description="Confidence score in sentiment classification"),
    ]
    
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    prompt_template = f"""
    Analyze the following news for {company_name} ({stock_code}):\n
    {chr(10).join(news_summaries)}\n
    Provide a JSON with keys: {', '.join([s.name for s in response_schemas])}
    """
    
    prompt = PromptTemplate(template=prompt_template, output_parser=parser)

    # llm = VertexAI(model_name="gemini-2.0-flash", temperature=0)
    # result = llm(prompt.format_prompt(news_summaries=news_summaries).to_string())
    
    client = genai.Client(http_options=HttpOptions(api_version='v1'))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt.format_prompt(news_summaries=news_summaries).to_string())
    print(response.text)
    return parser.parse(response.text)

def main(company_name: str):
    with mlflow.start_run(run_name=f"Analysis for {company_name}"):
        # Step 1: Stock code extraction
        stock_code = get_stock_code(company_name)
        mlflow.log_param("stock_code", stock_code)
        
        # Step 2: Fetch news and log metric with step info
        news = fetch_news(stock_code)
        mlflow.log_metric("news_count", len(news), step=1)
        
        # Step 3: Sentiment analysis and log confidence score with step info
        sentiment_profile = analyze_sentiment(news, company_name, stock_code)
        mlflow.log_metric("confidence_score", sentiment_profile['confidence_score'], step=2)
        
        return sentiment_profile

if __name__ == "__main__":
    company = "Google"
    output = main(company)
    print(output)
