# In-silico-Modellierung der zyklus- und kontrazeptionsabhängigen Glukose-Insulin-Dynamik

Interaktive Web-Applikation (Streamlit) sowie begleitendes Jupyter Notebook auf Basis des erweiterten **Bergman-Minimal-Modells (Oral Minimal Model)** zur Analyse der postprandialen Glukosedynamik und zellulären Insulinsensitivität ($S_I$) im Verlauf des natürlichen Menstruationszyklus sowie unter kombinierten oralen Kontrazeptiva (KOK).

## Wissenschaftliche Datengrundlage
- **Follikelphase & Lutealphase:** Bennal & Kerure (2013) – Standard-OGTT (75 g Glukose)
- **Kombinierte orale Kontrazeptiva (150 µg Levonorgestrel / 30 µg Ethinylestradiol):** Godsland et al. (1990, *NEJM*)
- **Externe Literatur-Validierung (IVGTT):** Escalante Pulido & Alpizar Salazar (1999)

## Starten der Web-App
```bash
pip install -r requirements.txt
streamlit run app.py
```