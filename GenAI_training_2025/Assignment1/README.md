# Assignment1
Real-Time Market Sentiment Analyzer using LangChain, Google Gemini-2.0-flash, and mlflow:

***

# Real-Time Market Sentiment Analyzer

## Overview

This project implements a LangChain-powered pipeline to analyze market sentiment for companies in real-time. It accepts a company name as input, extracts its stock ticker, fetches recent news, and uses Google Gemini-2.0-flash (via Vertex AI) to generate a structured sentiment profile. The pipeline includes comprehensive observability using mlflow for tracing, prompt debugging, and monitoring.

***

## Features

- Stock code extraction via Yahoo Finance Symbol Suggest tool
- News retrieval using Brave Search or other integrated news tools
- Sentiment analysis and named entity extraction using Google Gemini-2.0-flash
- Structured JSON output detailing sentiment and market implications
- Integrated mlflow tracking for prompt logging, metrics, and tracing spans

***

## Tech Stack

- Python 3.10+
- LangChain
- Google Gemini-2.0-flash model (Vertex AI)
- Yahoo Finance / Brave Search / Exa Search tools (LangChain integrations)
- mlflow for observability and tracing

***

## Running the Pipeline

Run the main script by passing a company name as argument or modifying the `main()` call:

```bash
python market_sentiment_analyzer.py --company "Google"
```






