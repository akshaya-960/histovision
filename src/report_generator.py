import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image


def generate_pdf_report(result, class_names):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "HistoVision AI — Analysis Report")
    y -= 20
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  File: {result['filename']}")
    c.setFillColor(colors.black)
    y -= 30

    c.setFont("Helvetica-Bold", 11)
    ood_text = "OUT-OF-DISTRIBUTION WARNING" if result["is_ood"] else "In-distribution (consistent with training data)"
    c.setFillColor(colors.red if result["is_ood"] else colors.green)
    c.drawString(50, y, ood_text)
    c.setFillColor(colors.black)
    y -= 15
    c.setFont("Helvetica", 9)
    c.drawString(50, y, f"Mahalanobis distance: {result['min_dist']:.2f}")
    y -= 30

    img_buf = io.BytesIO()
    Image.fromarray(result["img_resized"]).save(img_buf, format="PNG")
    img_buf.seek(0)

    resnet_overlay_buf = io.BytesIO()
    Image.fromarray(result["overlay"]).save(resnet_overlay_buf, format="PNG")
    resnet_overlay_buf.seek(0)

    vit_overlay_buf = io.BytesIO()
    Image.fromarray(result["vit_overlay"]).save(vit_overlay_buf, format="PNG")
    vit_overlay_buf.seek(0)

    img_size = 1.6 * inch
    gap = 15
    x = 50
    for buf, label in [(img_buf, "Original"), (resnet_overlay_buf, "Grad-CAM (ResNet18)"), (vit_overlay_buf, "Attn. Rollout (ViT)")]:
        c.drawImage(ImageReader(buf), x, y - img_size, width=img_size, height=img_size)
        c.setFont("Helvetica", 7)
        c.drawString(x, y - img_size - 10, label)
        x += img_size + gap
    y -= img_size + 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model Predictions")
    y -= 20

    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, y, "Class")
    c.drawString(180, y, "ResNet18")
    c.drawString(300, y, "ViT-B/16")
    y -= 12
    c.line(50, y, 420, y)
    y -= 12

    c.setFont("Helvetica", 9)
    for i, cls in enumerate(class_names):
        c.drawString(50, y, cls)
        c.drawString(180, y, f"{result['resnet_probs'][i]*100:.1f}%")
        c.drawString(300, y, f"{result['vit_probs'][i]*100:.1f}%")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 10)
    agree_text = "Models AGREE" if result["agree"] else "Models DISAGREE — recommend manual review"
    c.setFillColor(colors.green if result["agree"] else colors.red)
    c.drawString(50, y, agree_text)
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    c.drawString(50, 50, "Research prototype. Not validated for clinical use. Not a diagnostic tool.")

    c.save()
    buffer.seek(0)
    return buffer