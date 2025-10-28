# tools/weather_api.py
import requests
from langchain_community.tools import tool
from datetime import datetime

@tool
def get_weather_forecast(latitude: float, longitude: float) -> str:
    """
    Fetches the 7-day weather forecast for a specific latitude and longitude.
    Returns a formatted string with daily weather predictions including temperature,
    precipitation, and wind speed.
    """
    print(f"---TOOL: Fetching weather for Lat={latitude}, Lon={longitude}---")
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
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except (KeyError, IndexError) as e:
        return f"Error processing weather data: {e}"

