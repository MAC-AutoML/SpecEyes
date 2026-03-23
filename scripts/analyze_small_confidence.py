import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Configure plotting fonts.
plt.rcParams['axes.unicode_minus'] = False  # Keep minus signs visible.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'serif']

def analyze_confidence_distribution(json_file, output_dir):
    """Analyze confidence trends for a single judge result file."""

    font_size_label = 20
    font_size_title = 22
    font_size_tick = 18
    font_size_legend = 18
    
    # Load result data.
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read().replace('NaN', 'null')
        data = json.loads(content)

        for mode in ['min', 'mean', 'bottom20', 'log']:
            confidence_scores = []
            is_correct = []
            output_dir_per_mode = os.path.join(output_dir, mode)
            os.makedirs(output_dir_per_mode, exist_ok=True)
            conf_infos = data['conf_infos'][mode]
            if 'wrong' in conf_infos:
                for score in conf_infos['wrong']:
                    confidence_scores.append(score)
                    is_correct.append(0)

            if 'right' in conf_infos:
                for score in conf_infos['right']:
                    confidence_scores.append(score)
                    is_correct.append(1)

            if len(confidence_scores) == 0:
                return

            confidence_scores = np.array(confidence_scores)
            is_correct = np.array(is_correct)

            # Sort samples by confidence score.
            sorted_idx = np.argsort(confidence_scores)
            sorted_conf = confidence_scores[sorted_idx]
            sorted_correct = is_correct[sorted_idx]

            N = len(sorted_correct)
            x_axis = np.arange(N)

            w = max(10, N // 30)  # Use a window close to 3 percent of samples.
            rolling_acc = np.convolve(sorted_correct, np.ones(w)/w, mode='valid')

            plt.figure(figsize=(7,4))
            plt.plot(rolling_acc)
            plt.axhline(sorted_correct.mean(), linestyle='--', label='Global Acc')
            plt.xlabel("Confidence Rank", fontsize=font_size_label)
            plt.ylabel("Local Accuracy", fontsize=font_size_label)
            plt.title("Rolling Accuracy Trend", fontsize=font_size_title)
            plt.ylim(0, 1.05)  # Keep all accuracy plots on the same scale.
            plt.xticks(fontsize=font_size_tick)
            plt.yticks(fontsize=font_size_tick)
            plt.legend(fontsize=font_size_legend)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir_per_mode, 'rolling_accuracy.png'))
            plt.close()

            from scipy.stats import spearmanr
            rho, p = spearmanr(x_axis, sorted_correct)

            with open(os.path.join(output_dir_per_mode, 'stats.txt'), 'w') as f:
                f.write(f"Total samples: {N}\n")
                f.write(f"Global accuracy: {sorted_correct.mean():.4f}\n")
                f.write(f"Spearman correlation (rank vs correctness): {rho:.4f}\n")
                f.write(f"P-value: {p:.6e}\n")

            rev_correct = sorted_correct[::-1]
            cum_correct = np.cumsum(rev_correct)
            thresholds = []
            accuracies = []

            unique_conf = np.unique(sorted_conf)

            for tau in unique_conf:
                # Find the first position with confidence >= tau.
                idx = np.searchsorted(sorted_conf, tau, side='left')
                k = N - idx  # Number of samples with confidence >= tau.

                if k == 0:
                    continue

                acc = cum_correct[k-1] / k
                thresholds.append(tau)
                accuracies.append(acc)

            plt.figure(figsize=(7,4))
            plt.plot(thresholds, accuracies)
            plt.axhline(sorted_correct.mean(), linestyle='--', label='Global Acc')
            plt.xlabel("Confidence Threshold τ", fontsize=font_size_label)
            plt.ylabel("Accuracy (conf ≥ τ)", fontsize=font_size_label)
            plt.title("Accuracy vs Confidence Threshold", fontsize=font_size_title)
            plt.ylim(0, 1.05)
            plt.xticks(fontsize=font_size_tick)
            plt.yticks(fontsize=font_size_tick)
            plt.legend(fontsize=font_size_legend)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir_per_mode, 'threshold_accuracy.png'))
            plt.close()

            from scipy.stats import gaussian_kde

            right_conf = confidence_scores[is_correct == 1]
            wrong_conf = confidence_scores[is_correct == 0]

            if len(right_conf) > 1 and len(wrong_conf) > 1:
                x_range = np.linspace(confidence_scores.min(), confidence_scores.max(), 500)

                kde_right = gaussian_kde(right_conf)
                kde_wrong = gaussian_kde(wrong_conf)

                density_right = kde_right(x_range)
                density_wrong = kde_wrong(x_range)

                right_mean = np.mean(right_conf)
                wrong_mean = np.mean(wrong_conf)
                right_median = np.median(right_conf)
                wrong_median = np.median(wrong_conf)
                
                right_mode_idx = np.argmax(density_right)
                wrong_mode_idx = np.argmax(density_wrong)
                right_mode = x_range[right_mode_idx]
                wrong_mode = x_range[wrong_mode_idx]
                right_peak_height = density_right[right_mode_idx]
                wrong_peak_height = density_wrong[wrong_mode_idx]
                
                mean_distance = abs(right_mean - wrong_mean)
                median_distance = abs(right_median - wrong_median)
                mode_distance = abs(right_mode - wrong_mode)

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.fill_between(x_range, density_right, alpha=0.3, color='#2ecc71', label='Correct')
                ax.plot(x_range, density_right, color='#27ae60', linewidth=2.5, label='_nolegend_')

                ax.fill_between(x_range, density_wrong, alpha=0.3, color='#e74c3c', label='Incorrect')
                ax.plot(x_range, density_wrong, color='#c0392b', linewidth=2.5, label='_nolegend_')

                ax.axvline(right_mode, color='#27ae60', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.axvline(wrong_mode, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.8)
                
                max_density = max(right_peak_height, wrong_peak_height)
                arrow_y = max_density * 0.5
                ax.annotate('', xy=(wrong_mode, arrow_y), xytext=(right_mode, arrow_y),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
                mid_x = (right_mode + wrong_mode) / 2
                ax.text(mid_x, arrow_y + max_density * 0.08, f'Δ={mode_distance:.3f}', 
                        ha='center', va='bottom', fontsize=20, fontweight='bold')

                ax.set_xlabel("Confidence Score", fontsize=font_size_label)
                ax.set_ylabel("Density", fontsize=font_size_label)
                ax.set_title("Confidence Distribution (KDE)", fontsize=font_size_title)
                ax.set_xlim(x_range.min(), x_range.max())
                if mode == 'bottom20':
                    ax.set_xlim(x_range.min()-0.0025, x_range.max())
                if mode == 'min':
                    ax.set_xlim(x_range.min()+0.05, x_range.max())
                ax.tick_params(axis='both', which='major', labelsize=font_size_tick)
                ax.legend(fontsize=font_size_legend, framealpha=0.9, loc='upper left')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir_per_mode, 'confidence_kde.png'), dpi=300, bbox_inches='tight')
                plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze confidence trends from judge result files.")
    parser.add_argument("--input_folder", type=str, default="judge_results_qwen3vl-2b-Instruct")
    parser.add_argument("--output_root", type=str, default="conf_analysis")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_root = args.output_root
    
    # Create the output root directory.
    os.makedirs(output_root, exist_ok=True)
    
    # Iterate over all input files.
    for filename in os.listdir(input_folder):
        if filename.endswith('.jsonl'):
            input_file = os.path.join(input_folder, filename)
            
            # Use the file stem as the output subdirectory name.
            file_basename = os.path.splitext(filename)[0]
            output_dir = os.path.join(output_root, file_basename)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f'Processing file: {filename}')
            
            # Run the confidence analysis.
            analyze_confidence_distribution(input_file, output_dir)
            
            
            print(f'Analysis completed for {filename}')
            print(f'Results saved to: {output_dir}')

if __name__ == '__main__':
    main()
