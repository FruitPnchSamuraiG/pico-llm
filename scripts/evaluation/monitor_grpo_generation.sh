#!/bin/bash
# Monitor GRPO generation progress

OUTPUT_FILE="/scratch/kk6081/picollm_extend/gsm8k_grpo_groups_epoch5_s8_1k_v3.jsonl"
LOG_FILE="/scratch/kk6081/picollm_extend/grpo_gen_v3.log"

echo "🔍 Monitoring GRPO Generation Progress"
echo "========================================"
echo ""

while true; do
    clear
    echo "🔍 GRPO Generation Monitor"
    echo "========================================"
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    
    # Check if process is running
    if ps aux | grep -q "[g]enerate_grpo_groups_multi_gpu.py"; then
        echo "✅ Process: RUNNING"
        CPU=$(ps aux | grep "[g]enerate_grpo_groups_multi_gpu.py" | awk '{print $3}')
        MEM=$(ps aux | grep "[g]enerate_grpo_groups_multi_gpu.py" | awk '{print $4}')
        TIME=$(ps aux | grep "[g]enerate_grpo_groups_multi_gpu.py" | awk '{print $10}')
        echo "   CPU: ${CPU}% | Memory: ${MEM}% | Time: ${TIME}"
    else
        echo "❌ Process: NOT RUNNING"
    fi
    echo ""
    
    # Check output file
    if [ -f "$OUTPUT_FILE" ]; then
        SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        LINES=$(wc -l < "$OUTPUT_FILE")
        echo "📝 Output file: $SIZE ($LINES problems)"
        echo ""
        echo "Last 3 entries:"
        tail -3 "$OUTPUT_FILE" | jq -r '.problem_id' 2>/dev/null | while read id; do
            echo "  - Problem #$id"
        done
    else
        echo "📝 Output file: Not created yet"
    fi
    echo ""
    
    # Show last few log lines
    echo "📋 Recent log:"
    tail -10 "$LOG_FILE" 2>/dev/null | sed 's/^/   /'
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    
    sleep 10
done
