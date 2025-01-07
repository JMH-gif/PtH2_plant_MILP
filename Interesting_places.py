import folium
import numpy as np
# Create a map centered on the place
my_map = folium.Map(location=[53.113712, -4.029927], zoom_start=12)

# Add a marker with the image and description
html_wales = """
<b>Place to Visit</b><br>
<img src="https://www.bing.com/th/id/OBTQ.BT54FE972EE2A83DBCB347EB4AFA838C42604CB21DB7FF9D66D68BDC5E15BC843E?qlt=90&pid=InlineBlock" width="200"><br>
A beautiful place with amazing scenery!
"""

html_iceland_lighthouse = """
<b>Place to Visit</b><br>
<img src="https://www.locationscout.net/iceland/47903-svoertuloft-lighthouse" width="200"><br>
A beautiful place with amazing scenery!
"""
html_valley_of_temples_sicillia = """https://www.bing.com/spotlight?spotlightid=ValleyOfTemplesSicily&msclkid=a8d196887b111cc83d59f1e3c255e920&q=valley%20of%20the%20temples%20sicily&carindexpill=0&carindeximg=0&textorimgcar=img&isfullscreen=true&carscrlimgv2=0&form=SLVFUL&ssid=09e8f885-427b-8850-947d-202e0287b9e8&trsid=NONE&chevclk=false"""


popup = folium.Popup(html_wales, max_width=300)
popup_iceland = folium.Popup(html_iceland_lighthouse, max_width=300)
popup_temple_sicilia= folium.Popup(html_valley_of_temples_sicillia, max_width=300)
# Add the marker
folium.Marker(
    [53.113712, -4.029927],  # Coordinates
    popup=popup,
    tooltip="Click for details"
).add_to(my_map)

# Add the marker
folium.Marker(
    [64.8667, -24.0333],  # Coordinates
    popup=popup_iceland,
    tooltip="Click for details"
).add_to(my_map)

# Add the marker
folium.Marker(
    [37.29028, 13.58639],  # Coordinates
    popup=popup_temple_sicilia,
    tooltip="Click for details"
).add_to(my_map)


# Save the map to an HTML file
my_map.save("map_with_image.html")
# Killarney Provincial Park in Ontario, Kanada
print("Map has been saved as 'map_with_image.html'. Open it in your browser to view.")

import numpy as np

# Provided inflation rates for 2010 to 2023
inflation_rates = [1.011, 1.0208, 1.0201, 1.015, 1.0091, 1.0051, 1.0049, 1.0173, 1.0145, 1.0014, 1.0307, 1.0687, 1.0595]

# Calculate the cumulative inflation factor
cumulative_inflation_factor = np.prod(inflation_rates)

# Display the result
print(cumulative_inflation_factor)
