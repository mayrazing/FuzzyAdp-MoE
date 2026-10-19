import matplotlib.pyplot as plt

# ── Data ───────────────────────────────────────────────────────
epochs = list(range(1, 29))

train_loss = [
    1.2530, 0.8999, 0.7892, 0.7184, 0.6731, 0.6339, 0.5963, 0.5636,
    0.5406, 0.5111, 0.4922, 0.4716, 0.4429, 0.4268, 0.4104, 0.3961,
    0.3816, 0.3661, 0.2435, 0.1985, 0.1786, 0.1682, 0.1552, 0.1475,
    0.1416, 0.1360, 0.1244, 0.1227,
]
val_loss = [
    0.9393, 0.7713, 0.6950, 0.6536, 0.6147, 0.5880, 0.5637, 0.5754,
    0.6036, 0.5304, 0.5218, 0.5323, 0.5843, 0.5703, 0.5793, 0.5545,
    0.6090, 0.6021, 0.4758, 0.4801, 0.4900, 0.4962, 0.5060, 0.5165,
    0.5253, 0.5318, 0.5242, 0.5331,
]
train_acc = [
    0.5994, 0.7015, 0.7346, 0.7573, 0.7711, 0.7837, 0.7956, 0.8046,
    0.8131, 0.8214, 0.8277, 0.8344, 0.8464, 0.8488, 0.8548, 0.8594,
    0.8633, 0.8686, 0.9131, 0.9287, 0.9368, 0.9388, 0.9450, 0.9479,
    0.9483, 0.9505, 0.9558, 0.9555,
]
val_acc = [
    0.6899, 0.7383, 0.7728, 0.7888, 0.7939, 0.8009, 0.8091, 0.8126,
    0.7995, 0.8210, 0.8214, 0.8284, 0.8184, 0.8206, 0.8212, 0.8278,
    0.8114, 0.8259, 0.8582, 0.8649, 0.8625, 0.8641, 0.8627, 0.8619,
    0.8627, 0.8647, 0.8649, 0.8639,
]

lr_decay_epochs = [18, 26]
early_stop_epoch = 28

# ── Shared annotation helper ───────────────────────────────────────────────
def add_markers(ax, y_lr_labels):
    for i, ep in enumerate(lr_decay_epochs):
        ax.axvline(x=ep, color='#FF9800', linewidth=1.5, linestyle='--', alpha=0.85,
                   label=f'LR Decay (epoch {ep})')
        ax.text(ep + 0.2, y_lr_labels[i],
                f'LR↓\n(ep{ep})', color='#FF9800', fontsize=8, va='top')
    ax.axvline(x=early_stop_epoch, color='#9C27B0', linewidth=1.5,
               linestyle=':', alpha=0.9, label=f'Early Stop (epoch {early_stop_epoch})')
    ax.text(early_stop_epoch + 0.2, y_lr_labels[2],
            'Early\nStop', color='#9C27B0', fontsize=8, va='top')

# ── Canvas ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Loss and Accuracy Curves During Training and Validation of the EfficientNet-B0", fontsize=15, fontweight='bold', y=1.01)

# ── Left: Loss ─────────────────────────────────────────────────
ax1.plot(epochs, train_loss, color='#2196F3', linewidth=2,
         marker='o', markersize=4, label='Train Loss')
ax1.plot(epochs, val_loss,   color='#F44336', linewidth=2,
         marker='s', markersize=4, label='Val Loss')
add_markers(ax1, y_lr_labels=[1.1, 0.55, 0.75])
ax1.set_xlabel('Epoch\n\n(a)', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_xlim(0.5, 29)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.4)

# ── Right: Accuracy ─────────────────────────────────────────────
ax2.plot(epochs, train_acc, color='#2196F3', linewidth=2,
         marker='o', markersize=4, label='Train Acc')
ax2.plot(epochs, val_acc,   color='#F44336', linewidth=2,
         marker='s', markersize=4, label='Val Acc')
add_markers(ax2, y_lr_labels=[0.63, 0.60, 0.63])
ax2.set_xlabel('Epoch\n\n(b)', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_xlim(0.5, 29)
ax2.set_ylim(0.55, 1.0)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Figure saved as training_curves.png")