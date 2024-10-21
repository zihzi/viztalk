from PIL import Image 
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph
from random import randrange
from text_generator import summary, description

def create_pdf(dataset,openai_key):
    pdf_file = f"{dataset}_summary.pdf"
    can = canvas.Canvas(pdf_file)
    text_summary = summary(openai_key)
    p = Paragraph(text_summary)

    # Set pdf background color
    A4_height = A4[1]
    A4_width = A4[0]    
    left_padding = 0
    bottom_padding = 0
    width = A4_width
    height = A4_height
    r = randrange(254,255)
    g = randrange(120,255)
    b = randrange(40,210)
    can.setFillColorRGB(r/255, g/255, b/255, 0.4)
    can.rect(left_padding, bottom_padding, width, height, fill=1)

    # Set pdf title
    can.setFont("Helvetica-Bold", 18)
    can.setFillColor("black")
    can.drawCentredString(300,750,f"""Summary of {dataset}""")

    p.wrapOn(can, 500, 100)
    p.drawOn(can, 50, 600)
    #  x,y for 4 images
    # x_start = [30, 315, 30, 315]
    # y_start = [390, 390, 120, 120]

    # x,y for 2 images
    x_start = [30, 30]
    y_start = [390, 120]

    # x,y for 2 descriptions
    x_start_desc = [50, 50]
    y_start_desc = [300, 40]

    for i in range(len(x_start)):
        image_path = f'image_{i}.png'
        text_description = description(image_path, openai_key)
        p = Paragraph(text_description)
        p.wrapOn(can, 500, 100)
        p.drawOn(can, x_start_desc[i], y_start_desc[i])
        image = Image.open(image_path)
        # im = image.resize((1250,800))
        image.save(f"data2poster_img/image_{i}.png")
        can.drawImage(f"data2poster_img/image_{i}.png", x_start[i], y_start[i],width=520,height=160)
    can.showPage()
    can.save()



