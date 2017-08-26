package com.dmitrybrant.mushroomhunter;

import android.support.annotation.Nullable;

public class MinStreakResult {

    private int length;
    @Nullable private Classifier.Recognition currentResult;
    private int currentStreakLength;

    public MinStreakResult(int length) {
        this.length = length;
    }

    public void add(@Nullable Classifier.Recognition result) {
        if (currentResult == null) {
            currentResult = result;
        }
        if (result == null || !result.getTitle().equals(currentResult.getTitle())) {
            currentResult = result;
            currentStreakLength = 1;
            return;
        }
        currentResult = result;
        currentStreakLength++;
    }

    public Classifier.Recognition get() {
        return currentStreakLength < length ? null : currentResult;
    }
}
