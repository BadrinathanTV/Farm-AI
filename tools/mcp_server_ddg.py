from fastmcp import FastMCP
from ddgs import DDGS
import httpx
from bs4 import BeautifulSoup

# Initialize FastMCP Server
mcp = FastMCP("DuckDuckGo Search & Weather Forecast")

@mcp.tool()
def web_search(query: str, max_results: int = 10) -> str:
    """
    Performs a web search using DuckDuckGo (Free, No API Key).
    Useful for getting current market prices, news, and general information.
    """
    print(f"--- DDG SEARCH: Searching for '{query}' (Region: US-EN) ---")
    results = DDGS().text(query, region="us-en", timelimit="d", max_results=max_results)
    
    if not results:
        return "No results found."

    formatted_results = []
    for r in results:
        formatted_results.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}")

    return "\n---\n".join(formatted_results)

@mcp.tool()
def fetch_page(url: str) -> str:
    """
    Fetches the text content of a webpage.
    Useful for reading the full details of a search result (e.g., specific price tables).
    """
    print(f"--- DDG FETCH: Fetching '{url}' ---")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/"
        }
        response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove scripts and styles
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        text = soup.get_text(separator="\n")
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit to first 5000 chars to avoid token limits
        return text[:5000]
        
    except Exception as e:
        return f"Failed to fetch page: {e}"

@mcp.tool()
def get_weather_forecast(latitude: float, longitude: float) -> str:
    """
    Fetches the 7-day weather forecast for a specific latitude and longitude.
    Returns a formatted string with daily weather predictions including temperature,
    precipitation, and wind speed.
    """
    import requests
    from datetime import datetime
    
    print(f"--- MCP WEATHER: Fetching for Lat={latitude}, Lon={longitude} ---")
    API_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
        "forecast_days": 7
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        daily_data = data['daily']
        formatted_forecast = "7-Day Weather Forecast:\n"
        for i in range(len(daily_data['time'])):
            date = datetime.strptime(daily_data['time'][i], '%Y-%m-%d').strftime('%A, %b %d')
            max_temp = daily_data['temperature_2m_max'][i]
            min_temp = daily_data['temperature_2m_min'][i]
            precip = daily_data['precipitation_sum'][i]
            wind = daily_data['wind_speed_10m_max'][i]
            
            formatted_forecast += (
                f"- {date}: Temp {min_temp}°C to {max_temp}°C, "
                f"Precipitation: {precip}mm, Wind: up to {wind} km/h\n"
            )
        return formatted_forecast
    except Exception as e:
        return f"Error fetching weather data: {e}"

if __name__ == "__main__":
    # Run as SSE Server on port 8000
    mcp.run(transport="streamable-http", port=8000)
