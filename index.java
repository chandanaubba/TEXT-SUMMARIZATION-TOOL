import java.util.*;

public class ArticleSummarizer {

    // Split text into sentences
    public static List<String> getSentences(String text) {
        return Arrays.asList(text.split("(?<=[.!?])\\s*"));
    }

    // Get word frequencies (basic term frequency)
    public static Map<String, Integer> getWordFrequencies(String text) {
        Map<String, Integer> freqMap = new HashMap<>();
        String[] words = text.toLowerCase().split("\\W+");

        for (String word : words) {
            if (!word.isEmpty()) {
                freqMap.put(word, freqMap.getOrDefault(word, 0) + 1);
            }
        }
        return freqMap;
    }

    // Score each sentence based on word frequency
    public static Map<String, Double> scoreSentences(List<String> sentences, Map<String, Integer> wordFreqs) {
        Map<String, Double> scores = new HashMap<>();

        for (String sentence : sentences) {
            String[] words = sentence.toLowerCase().split("\\W+");
            double score = 0.0;

            for (String word : words) {
                if (wordFreqs.containsKey(word)) {
                    score += wordFreqs.get(word);
                }
            }
            scores.put(sentence, score);
        }
        return scores;
    }

    // Generate summary by selecting top sentences
    public static String getSummary(String text, int numSentences) {
        List<String> sentences = getSentences(text);
        Map<String, Integer> wordFreqs = getWordFrequencies(text);
        Map<String, Double> sentenceScores = scoreSentences(sentences, wordFreqs);

        return sentenceScores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(numSentences)
                .map(Map.Entry::getKey)
                .reduce("", (s1, s2) -> s1 + " " + s2).trim();
    }

    // Test the summarizer
    public static void main(String[] args) {
        String article = "Artificial intelligence (AI) is transforming industries across the globe. "
                + "It is used in healthcare for diagnosis and treatment recommendations, in finance for fraud detection and algorithmic trading, "
                + "and in autonomous vehicles for real-time decision making. AI also powers personal assistants like Siri and Alexa, "
                + "and is becoming increasingly embedded in everyday life. Despite the many benefits, there are concerns about job displacement, "
                + "privacy, and the ethical use of AI. As the technology advances, governments and organizations are working on guidelines and regulations "
                + "to ensure responsible use.";

        System.out.println("Original Article:\n" + article);
        System.out.println("\nSummary:\n" + getSummary(article, 3));
    }
}
