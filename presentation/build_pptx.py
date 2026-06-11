# -*- coding: utf-8 -*-
"""Génère la présentation .pptx du projet de classification d'images de chiens."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")

NAVY   = RGBColor(0x16, 0x2A, 0x4A)
BLUE   = RGBColor(0x2E, 0x5E, 0x9E)
ORANGE = RGBColor(0xE3, 0x7B, 0x28)
GRAY   = RGBColor(0x3C, 0x3C, 0x3C)
LGRAY  = RGBColor(0x8A, 0x8A, 0x8A)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT  = RGBColor(0xF2, 0xF4, 0xF7)

PRESENTERS = {
    "Evrard":  RGBColor(0x2E, 0x5E, 0x9E),
    "Romain":  RGBColor(0xE3, 0x7B, 0x28),
    "Zouhair": RGBColor(0x3F, 0x8A, 0x5B),
}

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def fig(name):
    return os.path.join(FIG, name)


def rect(slide, l, t, w, h, color, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp


def txt(slide, l, t, w, h, text, size, color=GRAY, bold=False,
        align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri", italic=False):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0; tf.margin_right = 0; tf.margin_top = 0; tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
        r.font.color.rgb = color; r.font.name = font
    return tb


def fitted(slide, path, bl, bt, bw, bh):
    pic = slide.shapes.add_picture(path, bl, bt)
    scale = min(bw / pic.width, bh / pic.height)
    pic.width = int(pic.width * scale)
    pic.height = int(pic.height * scale)
    pic.left = int(bl + (bw - pic.width) / 2)
    pic.top = int(bt + (bh - pic.height) / 2)
    return pic


def bullets(slide, l, t, w, h, items, size=18, color=GRAY, gap=10):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, (txt_, accent) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run(); r.text = "›  "
        r.font.size = Pt(size); r.font.bold = True
        r.font.color.rgb = accent; r.font.name = "Calibri"
        r2 = p.add_run(); r2.text = txt_
        r2.font.size = Pt(size); r2.font.color.rgb = color; r2.font.name = "Calibri"
    return tb


def content_slide(title, presenter, section_no):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, SW, SH, WHITE)
    # top accent bar
    rect(s, 0, 0, SW, Inches(0.16), PRESENTERS[presenter])
    # title
    txt(s, Inches(0.6), Inches(0.42), Inches(9.6), Inches(0.9), title,
        26, NAVY, bold=True)
    # presenter tag (top-right)
    tag = rect(s, Inches(11.0), Inches(0.4), Inches(1.9), Inches(0.5),
               PRESENTERS[presenter])
    txt(s, Inches(11.0), Inches(0.4), Inches(1.9), Inches(0.5), presenter,
        14, WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # section number bottom-left
    txt(s, Inches(0.6), Inches(7.0), Inches(4), Inches(0.4),
        section_no, 11, LGRAY)
    return s


# ---------------------------------------------------------------- 1. TITLE
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, Inches(2.55), SW, Inches(0.06), ORANGE)
txt(s, Inches(1.0), Inches(1.4), Inches(11.3), Inches(1.1),
    "Classification d'images de chiens", 40, WHITE, bold=True,
    align=PP_ALIGN.CENTER)
txt(s, Inches(1.0), Inches(2.7), Inches(11.3), Inches(0.8),
    "D'un pipeline classique aux réseaux profonds", 22, RGBColor(0xCF, 0xDB, 0xEC),
    align=PP_ALIGN.CENTER, italic=True)
txt(s, Inches(1.0), Inches(4.15), Inches(11.3), Inches(0.6),
    "Evrard Lecureur   ·   Romain Ben   ·   Zouhair Saitout", 20, WHITE,
    bold=True, align=PP_ALIGN.CENTER)
txt(s, Inches(1.0), Inches(5.5), Inches(11.3), Inches(1.2),
    "Polytech Nice Sophia — MAM3 — Introduction au Machine Learning\n"
    "Encadrants : Mahmoud Elsawy & Jean-Luc Bouchot (INRIA)   ·   Juin 2026",
    14, RGBColor(0xAE, 0xB9, 0xCC), align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------- 2. PLAN
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, WHITE)
rect(s, 0, 0, Inches(4.6), SH, LIGHT)
txt(s, Inches(0.6), Inches(0.7), Inches(3.6), Inches(1), "Le projet\nen bref",
    30, NAVY, bold=True)
txt(s, Inches(0.6), Inches(2.4), Inches(3.6), Inches(4),
    "Reconnaître la race\nd'un chien à partir\nd'une photo.\n\n6 races · 1 petit jeu\nde données (PASCAL VOC).",
    18, GRAY)
plan = [
    ("1.  Données & préparation des images", PRESENTERS["Evrard"], "Evrard"),
    ("2.  Caractéristiques : PCA & HOG", PRESENTERS["Romain"], "Romain"),
    ("3.  Classifieur kNN", PRESENTERS["Romain"], "Romain"),
    ("4.  Classifieur SVM", PRESENTERS["Zouhair"], "Zouhair"),
    ("5.  Limites & transfer learning", PRESENTERS["Zouhair"], "Zouhair"),
]
y = 1.0
for label, col, who in plan:
    rect(s, Inches(5.2), Inches(y), Inches(0.12), Inches(0.9), col)
    txt(s, Inches(5.5), Inches(y), Inches(6.2), Inches(0.9), label, 19, GRAY,
        bold=True, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, Inches(11.7), Inches(y), Inches(1.5), Inches(0.9), who, 13, col,
        bold=True, anchor=MSO_ANCHOR.MIDDLE)
    y += 1.12

# ============================================================ EVRARD
# 3. Données
s = content_slide("Les données : 6 races, classes équilibrées", "Evrard", "1 · Données")
bullets(s, Inches(0.6), Inches(1.5), Inches(5.6), Inches(4), [
    ("Jeu SmallDB tiré de PASCAL VOC", PRESENTERS["Evrard"]),
    ("6 races : Chihuahua, basset, Kerry blue\nterrier, groenendael, malinois, chow", PRESENTERS["Evrard"]),
    ("Un XML par photo = cadre du chien", PRESENTERS["Evrard"]),
    ("Classes équilibrées ?  Entropie de\nShannon  H = 1,786  ≈  ln(6) = 1,792", PRESENTERS["Evrard"]),
    ("→ on peut juger au simple taux\nde bonnes réponses", PRESENTERS["Evrard"]),
], size=18, gap=14)
fitted(s, fig("class_balance.png"), Inches(6.5), Inches(1.7),
       Inches(6.3), Inches(4.6))

# 4. Découpage
s = content_slide("Préparation 1 : découper autour du chien", "Evrard", "1 · Données")
fitted(s, fig("bounding_box_crop.png"), Inches(0.6), Inches(1.6),
       Inches(8.3), Inches(5.3))
bullets(s, Inches(9.2), Inches(2.2), Inches(3.6), Inches(4), [
    ("Beaucoup de décor parasite\nsur les photos brutes", PRESENTERS["Evrard"]),
    ("Le cadre XML sert à recadrer\nsur le chien", PRESENTERS["Evrard"]),
    ("On garde l'essentiel,\non enlève le bruit de fond", PRESENTERS["Evrard"]),
], size=17, gap=16)

# 5. Redimensionnement / padding
s = content_slide("Préparation 2 : même taille, sans déformer", "Evrard", "1 · Données")
fitted(s, fig("preprocessing_comparaisonV2.png"), Inches(0.6), Inches(1.6),
       Inches(8.3), Inches(5.3))
bullets(s, Inches(9.2), Inches(2.0), Inches(3.6), Inches(4.5), [
    ("Cible : 256×256 px", PRESENTERS["Evrard"]),
    ("On garde les proportions\n(pas d'écrasement)", PRESENTERS["Evrard"]),
    ("Bords vides à remplir :\nnoir / blanc / continu", PRESENTERS["Evrard"]),
    ("On garde le remplissage\ncontinu : pas de faux\ncontour pour le HOG", PRESENTERS["Evrard"]),
], size=17, gap=14)

# ============================================================ ROMAIN
# 6. PCA reconstruction
s = content_slide("PCA : résumer l'image sur peu d'axes", "Romain", "2 · Caractéristiques")
fitted(s, fig("pca_reconstruction.png"), Inches(0.6), Inches(1.55),
       Inches(7.0), Inches(5.4))
fitted(s, fig("pca_approximatrion.png"), Inches(7.7), Inches(1.55),
       Inches(5.2), Inches(3.0))
bullets(s, Inches(7.8), Inches(4.9), Inches(5.0), Inches(2.2), [
    ("256×256 = 65 536 valeurs → trop", PRESENTERS["Romain"]),
    ("90 % de l'info sur très peu d'axes", PRESENTERS["Romain"]),
    ("Reconstruction : flou → net quand k ↑", PRESENTERS["Romain"]),
], size=15, gap=8)

# 7. PCA scatter — message clé
s = content_slide("PCA : les races sont déjà mélangées", "Romain", "2 · Caractéristiques")
fitted(s, fig("pca_scatter_plots.png"), Inches(0.6), Inches(1.6),
       Inches(7.4), Inches(5.3))
rect(s, Inches(8.3), Inches(2.3), Inches(4.4), Inches(3.0), LIGHT)
txt(s, Inches(8.5), Inches(2.55), Inches(4.0), Inches(2.6),
    "Message clé\n\nSur les 2 premiers axes,\naucun groupe net :\nles races se chevauchent.\n\n→ la tâche sera difficile,\nquel que soit le classifieur.",
    18, NAVY, bold=False)
txt(s, Inches(8.5), Inches(2.55), Inches(4.0), Inches(0.5),
    "Message clé", 18, ORANGE, bold=True)

# 8. HOG
s = content_slide("HOG : décrire les contours", "Romain", "2 · Caractéristiques")
fitted(s, fig("sobel_hog.png"), Inches(0.6), Inches(1.6),
       Inches(8.0), Inches(5.3))
bullets(s, Inches(8.9), Inches(2.1), Inches(3.9), Inches(4), [
    ("La PCA voit l'allure générale,\npas les contours", PRESENTERS["Romain"]),
    ("HOG : directions des contours\npar petite case (filtres Sobel)", PRESENTERS["Romain"]),
    ("On concatène PCA + HOG :\nallure + contours", PRESENTERS["Romain"]),
], size=17, gap=16)

# 9. kNN analysis
s = content_slide("kNN : normalisation, biais-variance, features", "Romain", "3 · kNN")
fitted(s, fig("task6_analysis.png"), Inches(0.5), Inches(1.7),
       Inches(12.3), Inches(4.3))
b = slide_b = s
cols = [
    ("Normaliser est nécessaire : sinon\nles grands nombres PCA écrasent le HOG", Inches(0.6)),
    ("k petit = collé aux données,\nk grand = lissé mais moins précis", Inches(4.9)),
    ("PCA + HOG ensemble = meilleure\nerreur de test (gain petit mais réel)", Inches(9.0)),
]
for text, x in cols:
    txt(s, x, Inches(6.15), Inches(4.1), Inches(1.1), text, 14, GRAY,
        align=PP_ALIGN.CENTER)

# 10. kNN erreurs
s = content_slide("Où le kNN se trompe", "Romain", "3 · kNN")
fitted(s, fig("confusion_matrix_knn.png"), Inches(0.7), Inches(1.7),
       Inches(6.6), Inches(5.1))
bullets(s, Inches(7.7), Inches(2.3), Inches(5.1), Inches(4), [
    ("kNN de référence : k=5, PCA+HOG,\nnormalisé", PRESENTERS["Romain"]),
    ("Prédit trop souvent Kerry blue\net chow", PRESENTERS["Romain"]),
    ("Pas assez Chihuahua et\ngroenendael", PRESENTERS["Romain"]),
    ("Penche vers les races qui\noccupent le plus l'espace\ndes caractéristiques", PRESENTERS["Romain"]),
], size=17, gap=16)

# ============================================================ ZOUHAIR
# 11. SVM pipeline + stratégies
s = content_slide("SVM : pipeline automatisé + réglage", "Zouhair", "4 · SVM")
bullets(s, Inches(0.6), Inches(1.6), Inches(5.7), Inches(3), [
    ("Pipeline scikit-learn\n(Pipeline + FeatureUnion)", PRESENTERS["Zouhair"]),
    ("Découpage train/test cohérent :\ncosinus = 0,99995", PRESENTERS["Zouhair"]),
    ("GridSearchCV (CV 3 folds) :\nC=5, γ=1e-4, 15 axes PCA/race", PRESENTERS["Zouhair"]),
    ("Noyau RBF > linéaire (races\nnon séparables par des droites)", PRESENTERS["Zouhair"]),
], size=17, gap=12)
# table
tx, ty, tw = Inches(6.7), Inches(1.7), Inches(6.0)
rows = [
    ("Stratégie", "Test", True),
    ("OvO — linéaire", "53,78 %", False),
    ("OvR — linéaire", "50,60 %", False),
    ("OvO — RBF réglé", "55,78 %", False),
    ("OvR — RBF réglé", "57,77 %", "best"),
]
rh = Inches(0.78)
for i, (a, bcell, kind) in enumerate(rows):
    yy = ty + i * rh
    if kind is True:
        rect(s, tx, yy, tw, rh, NAVY)
        c = WHITE; bold = True
    elif kind == "best":
        rect(s, tx, yy, tw, rh, RGBColor(0xE9, 0xF2, 0xEC))
        c = PRESENTERS["Zouhair"]; bold = True
    else:
        rect(s, tx, yy, tw, rh, LIGHT if i % 2 else WHITE)
        c = GRAY; bold = False
    txt(s, tx + Inches(0.25), yy, Inches(3.8), rh, a, 17, c, bold=bold,
        anchor=MSO_ANCHOR.MIDDLE)
    txt(s, tx + Inches(4.0), yy, Inches(1.8), rh, bcell, 17, c, bold=bold,
        anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)

# 12. SVM confusion
s = content_slide("SVM : meilleure diagonale, confusions restantes", "Zouhair", "4 · SVM")
fitted(s, fig("confusion_matrix.png"), Inches(2.0), Inches(1.6),
       Inches(6.4), Inches(5.3))
bullets(s, Inches(8.7), Inches(2.4), Inches(4.1), Inches(3.5), [
    ("Diagonale un peu plus marquée\nque le kNN", PRESENTERS["Zouhair"]),
    ("Certaines races bien reconnues", PRESENTERS["Zouhair"]),
    ("Mais confusions persistantes\nentre races qui se ressemblent", PRESENTERS["Zouhair"]),
], size=17, gap=16)

# 13. Discussion — plafond
s = content_slide("Pourquoi on plafonne vers 55-58 %", "Zouhair", "5 · Limites")
rect(s, Inches(0.6), Inches(1.7), Inches(12.1), Inches(1.6), LIGHT)
txt(s, Inches(0.9), Inches(1.95), Inches(11.5), Inches(1.2),
    "Le problème n'est pas le classifieur ni le réglage : ce sont les CARACTÉRISTIQUES.",
    22, NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
bullets(s, Inches(0.9), Inches(3.6), Inches(11.4), Inches(3), [
    ("PCA et HOG décrivent des choses trop générales", PRESENTERS["Zouhair"]),
    ("Les races se jouent sur des détails fins : oreilles, museau, poil", PRESENTERS["Zouhair"]),
    ("On le voyait déjà : races mélangées dès le scatter PCA", PRESENTERS["Zouhair"]),
    ("Aucun réglage de k, C ou γ ne crée une séparation qui n'existe pas", PRESENTERS["Zouhair"]),
], size=19, gap=16)

# 14. Transfer learning — résultat
s = content_slide("Transfer learning : on fait sauter le plafond", "Zouhair", "5 · Limites")
bullets(s, Inches(0.6), Inches(1.55), Inches(6.0), Inches(3.2), [
    ("VGG16 pré-entraîné sur des millions\nd'images, couches gelées", PRESENTERS["Zouhair"]),
    ("Il remplace PCA / HOG pour calculer\nles caractéristiques", PRESENTERS["Zouhair"]),
    ("On ajoute une petite tête entraînée\nsur nos 6 races (images couleur 224×224)", PRESENTERS["Zouhair"]),
], size=16, gap=12)
# banner 57 -> 98
rect(s, Inches(0.6), Inches(5.0), Inches(6.0), Inches(1.5), NAVY)
txt(s, Inches(0.7), Inches(5.18), Inches(5.8), Inches(0.5),
    "PCA+HOG  →  VGG16", 16, ORANGE, bold=True, align=PP_ALIGN.CENTER)
txt(s, Inches(0.7), Inches(5.6), Inches(5.8), Inches(0.8),
    "57 %   →   98 %", 30, WHITE, bold=True, align=PP_ALIGN.CENTER)
fitted(s, fig("confusion_matrix_vgg16.png"), Inches(6.9), Inches(1.5),
       Inches(6.0), Inches(5.3))

# 15. Conclusion
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, Inches(1.7), SW, Inches(0.06), ORANGE)
txt(s, Inches(1.0), Inches(0.9), Inches(11.3), Inches(0.8), "Conclusion",
    34, WHITE, bold=True, align=PP_ALIGN.CENTER)
bullets_items = [
    "Chaîne complète : nettoyage → caractéristiques → classification → évaluation",
    "Idées clés : normalisation, compromis biais-variance, découpage équilibré",
    "La qualité des caractéristiques fixe la limite : ~57 % en classique",
    "Transfer learning (VGG16) confirme le diagnostic : 98 %",
]
tb = s.shapes.add_textbox(Inches(1.6), Inches(2.3), Inches(10.1), Inches(3.5))
tf = tb.text_frame; tf.word_wrap = True
for i, it in enumerate(bullets_items):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(18)
    r = p.add_run(); r.text = "›  "; r.font.bold = True
    r.font.size = Pt(21); r.font.color.rgb = ORANGE; r.font.name = "Calibri"
    r2 = p.add_run(); r2.text = it
    r2.font.size = Pt(21); r2.font.color.rgb = WHITE; r2.font.name = "Calibri"
txt(s, Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.8),
    "Merci de votre attention — des questions ?", 22, RGBColor(0xCF, 0xDB, 0xEC),
    bold=True, align=PP_ALIGN.CENTER, italic=True)

out = os.path.join(HERE, "presentation.pptx")
prs.save(out)
print("OK ->", out, "|", len(prs.slides.__iter__().__length_hint__() if hasattr(prs.slides, "__length_hint__") else []), "slides")
print("Slides:", len(prs.slides._sldIdLst))
