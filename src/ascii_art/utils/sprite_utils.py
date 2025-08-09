from PIL import Image, ImageFont, ImageDraw  

ascii_chars = "@%#*+:."
# ascii_chars = " "
edge_chars = "-/|\\"      
# creating a image object  
image = Image.new("L", (12*len(ascii_chars),20))

draw = ImageDraw.Draw(image)  
  
# specified font size 
font = ImageFont.truetype('/content/FiraCode-Regular.ttf', 20)  
  

  
# drawing text size 
draw.text((0,-3), ascii_chars, font = font,fill="white")  
  
image.show()
image.save('stacked_masks.png')