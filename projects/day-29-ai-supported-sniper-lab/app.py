import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import joblib

MODEL_BUNDLE = joblib.load("projectile_model.joblib")
AI_MODEL = MODEL_BUNDLE['model']
SCALER_X = MODEL_BUNDLE['scaler_X']
SCALER_Y = MODEL_BUNDLE['scaler_y']


def calculate_trajectory(v0, theta_degrees, g=9.81, num_steps=200):
    if v0 <= 0: return np.array([0]), np.array([0])
    theta_rad = np.deg2rad(theta_degrees)
    t_flight = (2 * v0 * np.sin(theta_rad)) / g
    t = np.linspace(0, t_flight, num_steps)
    x = v0 * np.cos(theta_rad) * t
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    return x, y

def create_plot(x_coords, y_coords, target=None, title="Trajectory"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_coords, y_coords, color='dodgerblue', linewidth=3, label='Path')
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    if target:
        ax.plot(target[0], target[1], 'X', color='red', markersize=15, label='Target')
    ax.set_xlabel("Menzil (metre)")
    ax.set_ylabel("Yükseklik (metre)")
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.tight_layout()
    return fig

def update_simulation_plot(v0, theta, g):
    x, y = calculate_trajectory(v0, theta, g)
    title = f"Simülasyon: v₀={v0:.1f} m/s, θ={theta:.1f}°"
    return create_plot(x, y, title=title)


with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky"), title="AI Nişancı") as demo:
    gr.Markdown("# 🎯 Yapay Zeka Destekli Nişancı Laboratuvarı")
    gr.Markdown("Fizik simülasyonu ile yapay zekanın hedefleri nasıl vurduğunu keşfedin.")

    with gr.Tabs():
        with gr.TabItem("🚀 Fizik Simülatörü", id=0):
            gr.Markdown("Parametreleri değiştirerek bir merminin yörüngesini canlı olarak gözlemleyin.")
            with gr.Row():
                with gr.Column(scale=1):
                    v0_slider = gr.Slider(20, 200, value=100, label="Başlangıç Hızı (v₀)", info="m/s")
                    theta_slider = gr.Slider(10, 80, value=45, label="Atış Açısı (θ)", info="derece")
                    g_slider = gr.Slider(1, 25, value=9.81, label="Yer Çekimi İvmesi (g)", info="m/s²")
                with gr.Column(scale=3):
                    sim_plot = gr.Plot()
            
            v0_slider.change(update_simulation_plot, inputs=[v0_slider, theta_slider, g_slider], outputs=sim_plot)
            theta_slider.change(update_simulation_plot, inputs=[v0_slider, theta_slider, g_slider], outputs=sim_plot)
            g_slider.change(update_simulation_plot, inputs=[v0_slider, theta_slider, g_slider], outputs=sim_plot)


        with gr.TabItem("🤖 Yapay Zeka Hedefleyici", id=1):
            gr.Markdown("Hedefin **X (Menzil)** ve **Y (Yükseklik)** koordinatlarını girin, ardından butona basın.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Hedef Koordinatları")
                    target_x_input = gr.Number(label="Hedef X (metre)", value=500)
                    target_y_input = gr.Number(label="Hedef Y (metre)", value=150)
                    fire_button = gr.Button("🎯 Hedefi Vur!", variant="primary")
                    gr.Markdown("### AI Tahminleri:")
                    v0_out = gr.Label(label="Gereken Hız (v₀)")
                    theta_out = gr.Label(label="Gereken Açı (θ)")
                    trajectory_info_out = gr.Label(label="Yörünge Analizi")
                with gr.Column(scale=3):
                    result_plot = gr.Plot(label="Sonuç Grafiği")

            def ai_targeter_on_button_click(target_x, target_y):
                if target_x is None or target_y is None or target_x <= 0 or target_y < 0:
                    blank_x, blank_y = np.array([0]), np.array([0])
                    fig = create_plot(blank_x, blank_y, title="Lütfen geçerli bir hedef girin!")
                    return fig, "Geçersiz Hedef", "Geçersiz Hedef", "Analiz Yok"

                input_scaled = SCALER_X.transform(np.array([[target_x, target_y]]))
                
                prediction_scaled = AI_MODEL.predict(input_scaled)
                predicted_params = SCALER_Y.inverse_transform(prediction_scaled)

                predicted_v0, predicted_theta = predicted_params[0]
                
                g = 9.81
                theta_rad = np.deg2rad(predicted_theta)
                t_peak = (predicted_v0 * np.sin(theta_rad)) / g
                t_target = target_x / (predicted_v0 * np.cos(theta_rad))
                
                if np.isclose(t_target, t_peak, atol=0.1):
                    trajectory_status = "🎯 Zirvede Vuruldu!"
                elif t_target < t_peak:
                    trajectory_status = "⬆️ Yükselirken Vuruldu!"
                else:
                    trajectory_status = "⬇️ Düşerken Vuruldu!"

                x_ai, y_ai = calculate_trajectory(predicted_v0, predicted_theta)
                title = f"AI Sonucu: v₀={predicted_v0:.1f} m/s, θ={predicted_theta:.1f}°"
                fig = create_plot(x_ai, y_ai, target=(target_x, target_y), title=title)
                
                v0_text = f"{predicted_v0:.2f} m/s"
                theta_text = f"{predicted_theta:.2f}°"
                
                return fig, v0_text, theta_text, trajectory_status

            fire_button.click(
                fn=ai_targeter_on_button_click, 
                inputs=[target_x_input, target_y_input], 
                outputs=[result_plot, v0_out, theta_out, trajectory_info_out]
            )
            
    demo.load(update_simulation_plot, inputs=[v0_slider, theta_slider, g_slider], outputs=sim_plot)

demo.launch()