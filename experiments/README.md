# Chunk Size Experiment

This experiment finds the best chunk size and overlap configuration for semantic search.

## How it works

1. **Load sample data** from `sample_es_data.json`
2. **Create multiple Qdrant collections** with different chunk configurations:
   - 128 tokens, 25 overlap
   - 256 tokens, 50 overlap  
   - 512 tokens, 100 overlap
   - 1024 tokens, 200 overlap
3. **Load test questions** from `test_questions.csv`
4. **Run searches** on each collection
5. **Calculate F1 scores** based on finding expected answers
6. **Find the best configuration**

## Usage

1. Make sure you have `sample_es_data.json` in the parent directory
2. Create `test_questions.csv` with your test questions:
   ```csv
   question,expected_answer,elastic_id
   وصی کیست؟,كسى را كه به او وصیت مى‏كنند «وصی» مى‏گويند.,14498
   ```
3. Run the experiment:
   ```bash
   cd experiments
   python find_best_chunk_size.py
   ```

## Test Questions CSV Format

The CSV file should have these columns:
- `question`: The search query
- `expected_answer`: The expected answer text
- `elastic_id`: The ID of the record that should be found

## Results

The script will show:
- Average F1 score for each configuration
- The best chunk size and overlap combination
- Detailed results for analysis

## Simple and Readable

The code follows these principles:
- No nested loops
- Clear function names
- Simple data structures
- Easy to understand flow
- Human-readable output


TODO: here
------------
باید یک دیتاست خوب برای پیدا کردن متن از متون بلند پیدا کنم

<!-- 1- باید نحوه جمع آوری سمپل هارو تغییر بدم
در حالت الان٫ میگرده به دنبال سمپل هایی که هم
سوال طولانی
هم جواب طولانی -->
DONE

<!-- ولی سوال مهم نیس
میتونه فقط جوابش طولانی باشه -->

DONE


<!-- بعدش باید بتونم توی یه جایی سوال جواب هارو دونه دونه بخونم
و سوال خوب ازشون طرح کنم -->
<!-- 
اوه پسر این خیلی طول کشید
مجبور شدم ترمینالم رو عوض کنم چون متن فارسی توش درست تایپ نمیشد -->

DONE


2- از توی اون جواب طولانی ها
باید بگردم٫‌جواب هایی که در بخشی از متن به موضوع خاصی میپردازن رو پیدا کنم
مثلا در ابتدای متن داره درباره نماز میگه
یهو وسطش روزه میگه
این متن خوبیه که باید ازش دیتاست تستم رو در بیارم

بعدم که رواله دیگه