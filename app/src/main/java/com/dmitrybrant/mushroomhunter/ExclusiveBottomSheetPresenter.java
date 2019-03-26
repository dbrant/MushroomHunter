package com.dmitrybrant.mushroomhunter;

import android.app.Dialog;
import androidx.annotation.NonNull;
import androidx.fragment.app.DialogFragment;
import androidx.fragment.app.FragmentManager;

public class ExclusiveBottomSheetPresenter {
    private static final String BOTTOM_SHEET_FRAGMENT_TAG = "bottom_sheet_fragment";
    private Dialog currentDialog;

    public void show(@NonNull FragmentManager manager, @NonNull DialogFragment dialog) {
        if (manager.isStateSaved() || manager.isDestroyed()) {
            return;
        }
        dismiss(manager);
        dialog.show(manager, BOTTOM_SHEET_FRAGMENT_TAG);
    }

    public void show(@NonNull FragmentManager manager, @NonNull Dialog dialog) {
        if (manager.isStateSaved() || manager.isDestroyed()) {
            return;
        }
        dismiss(manager);
        currentDialog = dialog;
        currentDialog.setOnDismissListener((dialogInterface) -> currentDialog = null);
        currentDialog.show();
    }

    public void dismiss(@NonNull FragmentManager manager) {
        if (manager.isStateSaved() || manager.isDestroyed()) {
            return;
        }
        DialogFragment dialog = (DialogFragment) manager.findFragmentByTag(BOTTOM_SHEET_FRAGMENT_TAG);
        if (dialog != null) {
            dialog.dismiss();
        }
        if (currentDialog != null) {
            currentDialog.setOnDismissListener(null);
            currentDialog.dismiss();
        }
        currentDialog = null;
    }
}
