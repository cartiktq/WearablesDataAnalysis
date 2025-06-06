#Agent which pings an online weather data service for weather information
class WeatherDataAgent:
    def __init__(self, agentic_client):
        self.client = agentic_client

    def execute(self):
        config = {"source": "noaa_or_openweather"}
        return self.client.query("weather", config)
