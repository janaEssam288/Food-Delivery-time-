# ==========================================
# 1. مكتبات التعامل مع البيانات والعمليات الحسابية
# ==========================================
import pandas as pd                      # للتعامل مع الجداول وتحليل البيانات (DataFrames)
import numpy as np                       # للعمليات الرياضية واستبدال القيم المفقودة (NaNs)

# ==========================================
# 2. مكتبات الرسم البياني والتصور (Visualization)
# ==========================================
import matplotlib.pyplot as plt          # وإنشاء الأشكال البيانية لتصور البيانات (EDA)
import seaborn as sns                    # مكتبة للتصور الإحصائي المتقدم والمبني على Matplotlib

# ==========================================
# 3. مكتبات بيئة Google Colab وملفات النظام
# ==========================================
import os                                # للتعامل مع مجلدات النظام ومسارات الملفات[cite: 2]
import zipfile                           # لفك ضغط الملفات المضغوطة (مثل archive.zip)[cite: 2]
from google.colab import files           # لرفع وتنزيل الملفات في بيئة جوجل كولاب[cite: 2]

# ==========================================
# 4. مكتبات التعلم الآلي والنمذجة (Scikit-Learn)
# ==========================================
from sklearn.model_selection import train_test_split  # لتقسيم البيانات إلى تدريب واختبار
from sklearn.compose import ColumnTransformer          # لتطبيق معالجة مختلفة على الأعمدة الرقمية والفئات[cite: 2]
from sklearn.pipeline import Pipeline                  # لدمج خطوات المعالجة والنموذج في خط إنتاج واحد[cite: 2]
from sklearn.impute import SimpleImputer              # لمعالجة القيم المفقودة (Imputation)
from sklearn.preprocessing import OneHotEncoder, StandardScaler # للترقيم والمعيرة
from sklearn.ensemble import RandomForestRegressor     # نموذج الغابات العشوائية للانحدار (Regression)[cite: 2]

# ==========================================
# 5. إعدادات خيارات العرض (Optional Settings)
# ==========================================
pd.set_option("display.max_columns", None) # عرض كل الأعمدة بدون اختصار[cite: 2]
pd.set_option("display.width", 160)        # ضبط عرض الشاشة للبيانات[cite: 2]