# tools/geocoding_api.py

import requests
from langchain_community.tools import tool

@tool
def get_coordinates_for_location(location_query: str) -> dict:
    """
    Fetches the latitude and longitude for a given location query (e.g., "Chennai, India").
    Returns a dictionary with 'latitude' and 'longitude' or an error message.
    """
    print(f"---TOOL: Geocoding for '{location_query}'---")
    # Using Nominatim (OpenStreetMap) - no API key needed, but be mindful of usage limits.
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': location_query, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'FarmAIAssistant/1.0'} # Nominatim requires a user-agent
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return {"latitude": lat, "longitude": lon}
        else:
            return {"error": "Location not found."}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}
    except (KeyError, IndexError, TypeError) as e:
        return {"error": f"Error parsing geocoding data: {e}"}
