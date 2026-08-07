import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import traceback

def test_eq(equation):
    eq_str = f"${equation}$"

    fig = plt.figure(figsize=(8, 2.5), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    try:
        ax.text(0.5, 0.5, eq_str, color='white', fontsize=28)
        plt.savefig('test_eq.png')
        print(f"Success: {equation}")
    except Exception as e:
        print(f"Error for {equation}:")
        traceback.print_exc()

test_eq(r"\ddot{\theta}_1 = f(\theta_1, \theta_2)")
test_eq(r"\frac{d^2\theta_1}{dt^2} = f(\theta_1, \theta_2)")
test_eq(r"\theta_{n+1} = \theta_n - \alpha \nabla f(\theta_n)")
