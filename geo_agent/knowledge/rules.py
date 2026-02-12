#!/usr/bin/env python3
"""
Geo-Agent 规则模块
实现 JGJ-106 桩基检测规则
"""

class JGJ106Rules:
    """
    JGJ-106 桩基检测规则类
    实现相关的判定规则
    """
    
    def __init__(self):
        """
        初始化规则引擎
        """
        # 基础阈值
        self.base_thresholds = {
            'intact': 0.95,      # 完整桩
            'minor_defect': 0.85,  # 轻微缺陷
            'major_defect': 0.70,  # 严重缺陷
            'broken': 0.50         # 断桩
        }
    
    def check_rule(self, beta, geo_type):
        """
        检查规则，根据地质类型调整阈值
        
        参数:
        - beta: 预测的置信度或其他指标
        - geo_type: 地质类型，如 'soft_soil' (软土), 'rock' (岩石), 'normal' (普通)
        
        返回:
        - 调整后的阈值
        """
        # 根据地质类型调整阈值
        if geo_type == 'soft_soil':
            # 软土：放宽阈值
            thresholds = {
                'intact': 0.90,
                'minor_defect': 0.80,
                'major_defect': 0.65,
                'broken': 0.45
            }
            print(f"🌍 地质类型: 软土，放宽判定阈值")
        elif geo_type == 'rock':
            # 岩石：收紧阈值
            thresholds = {
                'intact': 0.98,
                'minor_defect': 0.90,
                'major_defect': 0.75,
                'broken': 0.55
            }
            print(f"🌍 地质类型: 岩石，收紧判定阈值")
        else:
            # 普通地质：使用基础阈值
            thresholds = self.base_thresholds
            print(f"🌍 地质类型: 普通，使用标准判定阈值")
        
        return thresholds
    
    def classify_integrity(self, confidence, thresholds):
        """
        根据置信度和阈值分类桩身完整性
        
        参数:
        - confidence: 物理置信度
        - thresholds: 调整后的阈值
        
        返回:
        - 完整性等级和描述
        """
        if confidence >= thresholds['intact']:
            return 'intact', '桩身完整性良好'
        elif confidence >= thresholds['minor_defect']:
            return 'minor_defect', '桩身轻微缺陷'
        elif confidence >= thresholds['major_defect']:
            return 'major_defect', '桩身严重缺陷'
        else:
            return 'broken', '桩身断裂'
    
    def get_rule_explanation(self, geo_type):
        """
        获取规则解释
        
        参数:
        - geo_type: 地质类型
        
        返回:
        - 规则解释
        """
        explanations = {
            'soft_soil': "软土地质中，桩身周围土体较软，信号衰减较大，因此适当放宽判定阈值",
            'rock': "岩土地质中，桩身周围土体较硬，信号传播清晰，因此需要更严格的判定标准",
            'normal': "普通地质条件下，使用标准的判定阈值"
        }
        
        return explanations.get(geo_type, explanations['normal'])
    
    def validate_depth(self, defect_depth, pile_length):
        """
        验证缺陷深度是否合理
        
        参数:
        - defect_depth: 预测的缺陷深度
        - pile_length: 桩长
        
        返回:
        - 是否合理
        """
        if 0 < defect_depth < pile_length:
            return True
        return False
    
    def get_recommendation(self, integrity_level, defect_depth, pile_length):
        """
        根据完整性等级和缺陷深度给出建议
        
        参数:
        - integrity_level: 完整性等级
        - defect_depth: 缺陷深度
        - pile_length: 桩长
        
        返回:
        - 建议
        """
        if integrity_level == 'intact':
            return "桩身完整性良好，无需处理"
        elif integrity_level == 'minor_defect':
            return f"桩身在 {defect_depth:.1f}m 处存在轻微缺陷，建议进一步观察"
        elif integrity_level == 'major_defect':
            return f"桩身在 {defect_depth:.1f}m 处存在严重缺陷，建议进行补强处理"
        else:
            return f"桩身在 {defect_depth:.1f}m 处断裂，建议重新施工"

if __name__ == "__main__":
    # 测试规则引擎
    rules = JGJ106Rules()
    
    # 测试不同地质类型的阈值
    print("测试软土阈值:")
    soft_soil_thresholds = rules.check_rule(0.9, 'soft_soil')
    print(f"  软土阈值: {soft_soil_thresholds}")
    
    print("\n测试岩石阈值:")
    rock_thresholds = rules.check_rule(0.9, 'rock')
    print(f"  岩石阈值: {rock_thresholds}")
    
    print("\n测试普通地质阈值:")
    normal_thresholds = rules.check_rule(0.9, 'normal')
    print(f"  普通地质阈值: {normal_thresholds}")
    
    # 测试完整性分类
    print("\n测试完整性分类:")
    confidence = 0.92
    level, description = rules.classify_integrity(confidence, normal_thresholds)
    print(f"  置信度: {confidence}, 等级: {level}, 描述: {description}")
    
    # 测试建议生成
    print("\n测试建议生成:")
    recommendation = rules.get_recommendation('minor_defect', 8.5, 15.2)
    print(f"  建议: {recommendation}")
