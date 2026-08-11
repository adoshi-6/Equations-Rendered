from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import sys

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from renderer import load_font

def render_equation_latex_old(equation: str, output_path: str) -> bool:
    print(f"[DEBUG] Starting render_equation_latex for: {equation[:20]}...")
    try:
        # Matplotlib rendering
        fig = plt.figure(figsize=(10, 2), dpi=200)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_alpha(0.0)
        
        eq_str = equation.strip()
        if not eq_str.startswith('$'):
            eq_str = fr"${eq_str}$"
            
        ax.text(0.5, 0.5, eq_str, color='white', fontsize=28, 
                horizontalalignment='center', verticalalignment='center')
        
        plt.savefig(output_path, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print("[DEBUG] Successfully rendered equation via Matplotlib.")
        
    except Exception as e:
        print(f"[DEBUG] Matplotlib rendering failed: {e}")

    # 3. PIL Fallback
    try:
        print("[DEBUG] Entering PIL fallback block...")
        img = Image.new("RGBA", (1080, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = load_font(32)
            
        text_w = draw.textlength(equation, font=font)
        x_pos = (1080 - text_w) // 2
        print(f"[DEBUG] PIL fallback: string length is {len(equation)} chars, text_w is {text_w}, drawing at x={x_pos}")
        
        draw.text((x_pos, 80), equation, font=font, fill=(255, 255, 255, 255))
        img.save(output_path)
        print("[DEBUG] Successfully rendered equation via PIL. (THIS OVERWROTE THE MATPLOTLIB PNG!)")
        return True
    except Exception as e3:
        print(f"[DEBUG] PIL fallback rendering failed: {e3}")
        return False

print('--- DOUBLE PENDULUM ---')
eq_dp = r'\ddot{\theta}_1 = \frac{-g(2m_1+m_2)\sin\theta_1 - m_2 g\sin(\theta_1-2\theta_2) - 2\sin(\theta_1-\theta_2)m_2(\dot{\theta}_2^2 L_2 + \dot{\theta}_1^2 L_1 \cos(\theta_1-\theta_2))}{L_1(2m_1+m_2-m_2\cos(2\theta_1-2\theta_2))}'
render_equation_latex_old(eq_dp, 'dp_old.png')

print('\n--- ELECTRIC FIELD ---')
eq_ef = r'\vec{E} = \frac{1}{4\pi\epsilon_0} \sum_i q_i \frac{\vec{r} - \vec{r}_i}{|\vec{r} - \vec{r}_i|^3}'
render_equation_latex_old(eq_ef, 'ef_old.png')
