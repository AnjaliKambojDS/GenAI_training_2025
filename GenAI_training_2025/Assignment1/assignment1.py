import mlflow
from langchain.llms import VertexAI
from langchain.tools import YahooFinanceSymbolSuggestTool, BraveSearchTool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import SequentialChain, SimpleSequentialChain

def get_stock_code(company_name: str) -> str:
    
    tool = YahooFinanceSymbolSuggestTool()
    result = tool.run(company_name)
    # Extract and return stock ticker from result
    return result['symbol']

def fetch_news(stock_code: str) -> list[str]:
    
    tool = BraveSearchTool(max_results=5)
    news_results = tool.run(f"{stock_code} stock latest news")
    # Extract headlines or summaries
    return [item['headline'] for item in news_results]

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

    llm = VertexAI(model_name="gemini-2.0-flash", temperature=0)
    result = llm(prompt.format_prompt(news_summaries=news_summaries).to_string())
    return parser.parse(result)

def main(company_name: str):
    mlflow.start_run()
    
    with mlflow.start_run(step="stock_code_extraction"):
        stock_code = get_stock_code(company_name)
        mlflow.log_param("stock_code", stock_code)
    
    with mlflow.start_run(step="news_fetching"):
        news = fetch_news(stock_code)
        mlflow.log_metric("news_count", len(news))
    
    with mlflow.start_run(step="sentiment_analysis"):
        sentiment_profile = analyze_sentiment(news, company_name, stock_code)
        mlflow.log_metric("confidence_score", sentiment_profile['confidence_score'])
    
    mlflow.end_run()
    return sentiment_profile

if __name__ == "__main__":
    company = "Google"
    output = main(company)
    print(output)
