import cv2
import numpy as np
import os

def create_inference_test_data(num_images=50):
    base_dir = 'inference_test'
    
    # 목표 해상도 설정 (LR 기준)
    # x2: 1280x720 -> 640x360
    # x3: 1281x720 -> 427x240
    # x4: 1280x720 -> 320x180
    target_sizes = {
        'x2': (640, 360),
        'x3': (427, 240),
        'x4': (320, 180)
    }

    print("Inference 테스트용 데이터 생성 중...")

    for scale, (w, h) in target_sizes.items():
        save_path = os.path.join(base_dir, scale)
        os.makedirs(save_path, exist_ok=True)
        
        for i in range(1, num_images + 1):
            # 랜덤 노이즈 이미지 생성 (H, W, C)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            
            # 저장
            cv2.imwrite(os.path.join(save_path, f'test_{i:03d}.png'), img)
            
        print(f"[{scale}] {w}x{h} 이미지 {num_images}장 생성 완료.")

    print(f"\n모든 데이터가 '{base_dir}' 폴더에 준비되었습니다.")

if __name__ == "__main__":
    create_inference_test_data(50)