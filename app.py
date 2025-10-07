# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 09:39:34 2025

@author: admin
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# 设置页面
st.set_page_config(
    page_title="Medical Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Prognosis Risk of ICU patients with Viral Pneumonia Prediction System")
st.markdown("ExtraTrees Classifier for Medical Prognosis Prediction")

# 加载模型和特征
try:
    model = joblib.load('saved_models/medical_risk_model.pkl')
    
    # 直接从模型属性获取特征顺序（确保100%正确）
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        # 备用方案：从文件加载
        feature_names = joblib.load('saved_models/medical_risk_model_features.pkl')
    
    # 加载SHAP解释器
    try:
        explainer = joblib.load('saved_models/medical_risk_model_explainer.pkl')
    except:
        explainer = None
    
    st.success("✅ Model loaded successfully")
    
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# 创建输入字段 - 按照正确的特征顺序
input_values = {}

st.header("Patient Clinical Features")

col1, col2 = st.columns(2)

with col1:
    # 前6个特征
    for i, feature in enumerate(feature_names[:6]):
        if feature == 'apsiii':
            input_values[feature] = st.number_input("Acute Physiology Score III", min_value=0, max_value=300, value=50, key=feature)
        elif feature == 'resp_rate_min':
            input_values[feature] = st.number_input("Minimum Respiratory Rate", min_value=0.0, max_value=60.0, value=12.0, key=feature)
        elif feature == 'charlson_comorbidity_index':
            input_values[feature] = st.number_input("Charlson Comorbidity Index", min_value=0, max_value=20, value=2, key=feature)
        elif feature == 'bilirubin_total_min':
            input_values[feature] = st.number_input("Minimum Total Bilirubin", min_value=0.0, max_value=50.0, value=1.0, key=feature)
        elif feature == 'admission_age':
            input_values[feature] = st.number_input("Admission Age", min_value=0, max_value=120, value=65, key=feature)
        elif feature == 'aado2_calc_min':
            input_values[feature] = st.number_input("Minimum A-aDO2", min_value=0.0, max_value=800.0, value=100.0, key=feature)

with col2:
    # 后5个特征 - 特别注意顺序！
    for i, feature in enumerate(feature_names[6:], 7):
        if feature == 'height':
            input_values[feature] = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0, key=feature)
        elif feature == 'wbc_max':
            input_values[feature] = st.number_input("Maximum White Blood Cell Count", min_value=0.0, max_value=50.0, value=10.0, key=feature)
        elif feature == 'dbp_max':  # 注意：这个在第9位
            input_values[feature] = st.number_input("Maximum Diastolic Blood Pressure", min_value=0.0, max_value=200.0, value=80.0, key=feature)
        elif feature == 'pao2fio2ratio_min':  # 注意：这个在第10位
            input_values[feature] = st.number_input("Minimum PaO2/FiO2 Ratio", min_value=0.0, max_value=600.0, value=300.0, key=feature)
        elif feature == 'ptt_max':
            input_values[feature] = st.number_input("Maximum PTT", min_value=0.0, max_value=200.0, value=35.0, key=feature)

# 预测按钮
if st.button('Predict Mortality Risk', type="primary"):
    # 按照正确的特征顺序准备数据
    input_data = np.array([[input_values[feature] for feature in feature_names]])
    
    # 创建DataFrame
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # 进行预测
    try:
        # 获取预测概率
        probabilities = model.predict_proba(input_df)[0]
        mortality_prob = probabilities[1]  # 假设类别1是死亡
        
        # 显示结果
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric(
                '30-Day Mortality Probability', 
                f'{mortality_prob:.3f}',
                delta=f"{(mortality_prob-0.5)*100:+.1f}%" if mortality_prob > 0.5 else ""
            )
            
        with col2:
            # 风险等级
            if mortality_prob < 0.3:
                risk_level = "Low Risk"
                color = "#008bfa"
                emoji = "🔵"
            elif mortality_prob < 0.7:
                risk_level = "Medium Risk"
                color = "orange" 
                emoji = "🟡"
            else:
                risk_level = "High Risk"
                color = "#ff0050"
                emoji = "🔴"
                
            st.markdown(
                f"<h3 style='color:{color}'>{emoji} {risk_level}</h3>", 
                unsafe_allow_html=True
            )
        
        with col3:
            # 生存概率
            survival_prob = 1 - mortality_prob
            st.metric('Survival Probability', f'{survival_prob:.3f}')
        
        # 可视化部分 - 将两个图表放在一行，大小相同
        st.subheader("📊 Visualization")
        
        col1, col2 = st.columns(2)
        
        # 统一图表尺寸
        fig_size = (8, 6)  # 相同的图表尺寸
        
        with col1:
            st.markdown("**Probability Distribution**")
            # 饼图
            fig1, ax1 = plt.subplots(figsize=fig_size)
            labels = ['Survival', 'Mortality']
            sizes = [survival_prob, mortality_prob]
            colors = ['#008bfa', '#ff0050']
            explode = (0, 0.1)
            
            wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                             autopct='%1.1f%%', shadow=True, startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax1.axis('equal')
            ax1.set_title('Probability Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            st.markdown("**SHAP Feature Explanation**")
            if explainer is not None:
                try:
                    # 计算SHAP值
                    shap_values = explainer.shap_values(input_df)
                    
                    # 处理SHAP值的不同格式
                    if isinstance(shap_values, list):
                        # 列表格式：通常 [生存类SHAP值, 死亡类SHAP值]
                        if len(shap_values) == 2:
                            # 二分类情况 - 使用死亡类别的SHAP值
                            shap_array = shap_values[1][0]  # [1] 表示死亡类别, [0] 表示第一个样本
                        else:
                            # 其他列表格式，取第一个元素
                            shap_array = shap_values[0][0]
                    else:
                        # 数组格式
                        shap_array = shap_values[0]  # 第一个样本
                    
                    # 确保shap_array是1D
                    shap_array = np.array(shap_array).flatten()
                    
                    # 处理长度不匹配的情况
                    if len(shap_array) == 2 * len(feature_names):
                        # 如果是两倍长度，可能是两个类别的值拼接在一起了
                        # 取后一半作为死亡类别的SHAP值
                        shap_array = shap_array[len(feature_names):]
                    elif len(shap_array) != len(feature_names):
                        # 如果长度仍然不匹配，使用前n个值
                        shap_array = shap_array[:len(feature_names)]
                    
                    # 创建SHAP数据
                    shap_data = []
                    for i, feature in enumerate(feature_names):
                        if i < len(shap_array):
                            shap_val = float(shap_array[i])
                        else:
                            shap_val = 0.0  # 默认值
                        
                        impact = "Increases Risk" if shap_val > 0 else "Decreases Risk"
                        shap_data.append({
                            'Feature': feature,
                            'SHAP_Value': shap_val,
                            'Impact': impact,
                            'Absolute_Impact': abs(shap_val)
                        })
                    
                    shap_df = pd.DataFrame(shap_data)
                    shap_df_sorted = shap_df.sort_values('Absolute_Impact', ascending=True)
                    
                    # SHAP水平条形图
                    fig2, ax2 = plt.subplots(figsize=fig_size)
                    
                    colors_barh = ['#ff0050' if x > 0 else '#008bfa' for x in shap_df_sorted['SHAP_Value']]
                    bars = ax2.barh(shap_df_sorted['Feature'], shap_df_sorted['SHAP_Value'], 
                                    color=colors_barh, alpha=0.7)
                    
                    ax2.set_xlabel('SHAP Value', fontsize=12)
                    ax2.set_ylabel('Features', fontsize=12)
                    ax2.set_title('Feature Impact on Mortality Risk', fontsize=14, fontweight='bold')
                    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    
                    # 调整字体大小
                    ax2.tick_params(axis='both', which='major', labelsize=10)
                    
                    # 智能调整横坐标范围
                    shap_min = min(shap_df_sorted['SHAP_Value'])
                    shap_max = max(shap_df_sorted['SHAP_Value'])
                    
                    # 计算合适的横坐标范围
                    if abs(shap_min) < 0.01 and abs(shap_max) < 0.01:
                        # 如果SHAP值都很小，使用固定范围
                        x_range = [-0.02, 0.02]
                    else:
                        # 根据实际数据范围调整，留出边距
                        margin = max(abs(shap_min), abs(shap_max)) * 0.2
                        x_range = [shap_min - margin, shap_max + margin]
                    
                    # 确保范围不会太小
                    if x_range[1] - x_range[0] < 0.05:
                        center = (x_range[0] + x_range[1]) / 2
                        x_range = [center - 0.025, center + 0.025]
                    
                    ax2.set_xlim(x_range)
                    
                    # 添加数值标签 - 放在柱条另一侧并靠近中线
                    for bar in bars:
                        width = bar.get_width()
                        if abs(width) > 0.001:  # 只显示有显著影响的标签
                            # 所有数值标签都放在柱条的另一侧（远离中线的一侧）
                            if width >= 0:
                                # 正数：标签放在柱条右侧（远离中线）
                                label_x_pos = width + (x_range[1] - width) * 0.1  # 在柱条右侧10%位置
                                ha = 'left'  # 左对齐
                                color = 'darkred'  # 深红色文字
                                bg_color = 'lightcoral'  # 浅红色背景
                            else:
                                # 负数：标签放在柱条左侧（远离中线）
                                label_x_pos = width - (width - x_range[0]) * 0.1  # 在柱条左侧10%位置
                                ha = 'right'  # 右对齐
                                color = 'darkblue'  # 深蓝色文字
                                bg_color = 'lightblue'  # 浅蓝色背景
                            
                            # 确保标签位置合理
                            if width >= 0 and label_x_pos < width:
                                label_x_pos = width + 0.01  # 确保在柱条右侧
                            elif width < 0 and label_x_pos > width:
                                label_x_pos = width - 0.01  # 确保在柱条左侧
                            
                            ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                                    f'{width:.3f}', ha=ha, va='center', 
                                    fontweight='bold', fontsize=9, color=color,
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color, 
                                             edgecolor=color, alpha=0.8))
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    # 显示基础值
                    try:
                        if hasattr(explainer, 'expected_value'):
                            if isinstance(explainer.expected_value, (list, np.ndarray)):
                                if len(explainer.expected_value) == 2:
                                    base_val = float(explainer.expected_value[1])  # 死亡类别的基准值
                                else:
                                    base_val = float(explainer.expected_value[0])
                            else:
                                base_val = float(explainer.expected_value)
                            
                            st.markdown(f"**Base Value (Average Risk):** {base_val:.3f}")
                    except:
                        pass
                    
                except Exception as shap_error:
                    st.warning(f"SHAP explanation unavailable: {str(shap_error)}")
                    # 回退到标准特征重要性
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=True)
                        
                        fig2, ax2 = plt.subplots(figsize=fig_size)
                        bars = ax2.barh(importance_df['Feature'], importance_df['Importance'])
                        ax2.set_xlabel('Feature Importance', fontsize=12)
                        ax2.set_ylabel('Features', fontsize=12)
                        ax2.set_title('Model Feature Importance Ranking', fontsize=14, fontweight='bold')
                        
                        # 设置特征重要性的横坐标范围
                        imp_min = min(importance_df['Importance'])
                        imp_max = max(importance_df['Importance'])
                        imp_range = [imp_min - 0.02, imp_max + 0.02]
                        ax2.set_xlim(imp_range)
                        
                        # 标准特征重要性的数值标签放在柱条右侧
                        for bar in bars:
                            width = bar.get_width()
                            if width > 0.001:  # 只显示有显著影响的标签
                                label_x_pos = width + (imp_range[1] - width) * 0.1  # 在柱条右侧10%位置
                                ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                                       f'{width:.3f}', ha='left', va='center', 
                                       fontweight='bold', fontsize=9,
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', 
                                                edgecolor='black', alpha=0.8))
                        
                        plt.tight_layout()
                        st.pyplot(fig2)
            else:
                st.warning("SHAP explainer not available.")
                # 显示标准特征重要性
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig2, ax2 = plt.subplots(figsize=fig_size)
                    bars = ax2.barh(importance_df['Feature'], importance_df['Importance'])
                    ax2.set_xlabel('Feature Importance', fontsize=12)
                    ax2.set_ylabel('Features', fontsize=12)
                    ax2.set_title('Model Feature Importance Ranking', fontsize=14, fontweight='bold')
                    
                    # 设置特征重要性的横坐标范围
                    imp_min = min(importance_df['Importance'])
                    imp_max = max(importance_df['Importance'])
                    imp_range = [imp_min - 0.02, imp_max + 0.02]
                    ax2.set_xlim(imp_range)
                    
                    # 标准特征重要性的数值标签放在柱条右侧
                    for bar in bars:
                        width = bar.get_width()
                        if width > 0.001:  # 只显示有显著影响的标签
                            label_x_pos = width + (imp_range[1] - width) * 0.1  # 在柱条右侧10%位置
                            ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.3f}', ha='left', va='center', 
                                   fontweight='bold', fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', 
                                            edgecolor='black', alpha=0.8))
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
        
        # SHAP解释说明
        st.info("""
        **SHAP Value Interpretation:**
        - **🔴 Positive values (Red)**: Features that **increase** mortality risk
        - **🔵 Negative values (Blue)**: Features that **decrease** mortality risk  
        - **Value magnitude**: How strongly the feature affects the prediction
        - **Base value**: Average prediction across all patients
        """)
        
        # 显示详细的SHAP值表格
        if explainer is not None:
            try:
                st.subheader("📋 Detailed SHAP Values")
                detailed_shap_df = shap_df[['Feature', 'SHAP_Value', 'Absolute_Impact', 'Impact']].sort_values('Absolute_Impact', ascending=False)
                detailed_shap_df.columns = ['Feature', 'SHAP Value', 'Absolute Impact', 'Effect']
                st.dataframe(detailed_shap_df, use_container_width=True)
            except:
                pass
        
    except Exception as e:
        st.error(f'Prediction Error: {str(e)}')

# 页脚
st.markdown("---")

st.markdown("**Note**: This is a demonstration application. For clinical use, rigorous validation is required.")

