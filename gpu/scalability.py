import matplotlib.pyplot as plt

# Data
size = [1000000, 1000000000, 5000000000, 1000000000000]
op3_time = [10.0039, 1212.34, 90202, 664272]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(size, op3_time, marker='o', color='b', label='op3 time')

# Set labels and title
plt.xlabel('Size')
plt.ylabel('Time (ms)')
plt.title('Operation 3 Time vs Size')

# Set log scale for x-axis for better readability (optional)
plt.xscale('log')

# Display grid
plt.grid(True)

# Save the plot
plt.legend()
plt.savefig('op3_time_vs_size.png', format='png')

# Close the plot to free memory
plt.close()
