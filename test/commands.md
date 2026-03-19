curl -X POST "http://localhost:8000/aeo/page" \
  -H "Content-Type: application/json" \
  -d '{
    "entity": {
      "name": "Immortel AI",
      "oneLiner": "Next-gen AI agents",
      "website": "https://example.com",
      "offerings": ["Autonomous document generation"],
      "differentiators": ["LangGraph-based agents"],
      "competitors": ["Jasper AI"]
    },
    "intelligence": {
      "product_docs": "Immortel AI uses LangGraph...",
      "market_research": "Jasper focuses on marketing copy..."
    },
    "query": "immortel ai vs jasper ai for enterprise document automation",
    "page_type": "COMPARISON",
    "base_url": "https://example.com/aeo",
    "same_as_links": [
      "https://www.linkedin.com/company/immortel-ai"
    ]
  }'