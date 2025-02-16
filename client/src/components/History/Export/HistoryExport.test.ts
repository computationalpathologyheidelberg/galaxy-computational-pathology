import { createTestingPinia } from "@pinia/testing";
import { getLocalVue } from "@tests/jest/helpers";
import { shallowMount } from "@vue/test-utils";
import flushPromises from "flush-promises";
import { setActivePinia } from "pinia";

import {
    EXPIRED_STS_DOWNLOAD_RECORD,
    FILE_SOURCE_STORE_RECORD,
    RECENT_STS_DOWNLOAD_RECORD,
} from "@/components/Common/models/testData/exportData";
import type { components } from "@/schema";
import { mockFetcher } from "@/schema/__mocks__";
import { type HistorySummary, useHistoryStore } from "@/stores/historyStore";

import { getExportRecords } from "./services";

import HistoryExport from "./HistoryExport.vue";

const localVue = getLocalVue(true);

jest.mock("@/schema");
jest.mock("./services");
const mockGetExportRecords = getExportRecords as jest.MockedFunction<typeof getExportRecords>;
mockGetExportRecords.mockResolvedValue([]);

const FAKE_HISTORY_ID = "fake-history-id";
const FAKE_HISTORY = {
    id: FAKE_HISTORY_ID,
    name: "fake-history-name",
};

const REMOTE_FILES_API_ENDPOINT = new RegExp("/api/remote_files/plugins");

type FilesSourcePluginList = components["schemas"]["FilesSourcePlugin"][];
const REMOTE_FILES_API_RESPONSE: FilesSourcePluginList = [
    {
        id: "test-posix-source",
        type: "posix",
        label: "TestSource",
        doc: "For testing",
        writable: true,
        browsable: true,
        requires_roles: undefined,
        requires_groups: undefined,
    },
];

async function mountHistoryExport() {
    const pinia = createTestingPinia();
    setActivePinia(pinia);
    const historyStore = useHistoryStore(pinia);

    // the mocking method described in the pinia docs does not work in vue2
    // this is a work-around
    jest.spyOn(historyStore, "getHistoryById").mockImplementation(
        (_history_id: string) => FAKE_HISTORY as HistorySummary
    );

    const wrapper = shallowMount(HistoryExport, {
        propsData: { historyId: FAKE_HISTORY_ID },
        localVue,
        pinia,
    });
    await flushPromises();
    return wrapper;
}

describe("HistoryExport.vue", () => {
    beforeEach(async () => {
        mockFetcher.path(REMOTE_FILES_API_ENDPOINT).method("get").mock({ data: [] });
    });

    it("should render the history name", async () => {
        const wrapper = await mountHistoryExport();

        expect(wrapper.find("#history-name").text()).toBe(FAKE_HISTORY.name);
    });

    it("should render export options", async () => {
        const wrapper = await mountHistoryExport();

        expect(wrapper.find("#history-export-options").exists()).toBe(true);
    });

    it("should display a message indicating there are no exports where there are none", async () => {
        const wrapper = await mountHistoryExport();

        expect(wrapper.find("#no-export-records-alert").exists()).toBe(true);
    });

    it("should render previous records when there is more than one record", async () => {
        mockGetExportRecords.mockResolvedValue([
            RECENT_STS_DOWNLOAD_RECORD,
            FILE_SOURCE_STORE_RECORD,
            EXPIRED_STS_DOWNLOAD_RECORD,
        ]);
        const wrapper = await mountHistoryExport();

        expect(wrapper.find("#previous-export-records").exists()).toBe(true);
    });

    it("should not render previous records when there is one or less records", async () => {
        mockGetExportRecords.mockResolvedValue([RECENT_STS_DOWNLOAD_RECORD]);
        const wrapper = await mountHistoryExport();

        expect(wrapper.find("#previous-export-records").exists()).toBe(false);
    });

    it("should display file sources tab if there are available", async () => {
        mockFetcher.path(REMOTE_FILES_API_ENDPOINT).method("get").mock({ data: REMOTE_FILES_API_RESPONSE });
        const wrapper = await mountHistoryExport();

        expect(wrapper.find("#direct-download-tab").exists()).toBe(true);
        expect(wrapper.find("#file-source-tab").exists()).toBe(true);
    });

    it("should not display file sources tab if there are no file sources available", async () => {
        const wrapper = await mountHistoryExport();

        expect(wrapper.find("#direct-download-tab").exists()).toBe(true);
        expect(wrapper.find("#file-source-tab").exists()).toBe(false);
    });
});
