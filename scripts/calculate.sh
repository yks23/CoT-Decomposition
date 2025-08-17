for d in ./result/*/; do
    python -m preverification.cal --result_path "result/$(basename "$d")/result.json"
done
